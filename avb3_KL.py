import matplotlib.pyplot as plt
import numpy as np
import mdtraj
import keras
from deep_boltzmann.networks.training import MLTrainer, FlexibleTrainer
from deep_boltzmann.networks.invertible import invnet, EnergyInvNet
from deep_boltzmann.models.openmm import OpenMMEnergy
import mdtraj as md
from simtk import openmm, unit
import sys, os, shutil
import time
import tensorflow as tf
from avb3 import *

#リスタート用のステージ開始番号の辞書を作成
# NOTE: GPUのメモリオーバーで"Resource exhaustedエラー"がでた場合にはバッチ数を下げて、ステージ数を増やす
# 途中で学習が中断しときは、"中断したステージ番号"を以下に指定すれば、そのステージから学習をリスタート可能
restart = True
weight_file_name = "avb3_ML_stage5_saved.pkl"     # restart=Falseのとき、ML学習で保存した重みデータ名を与える
inter_model = False # 各ステージ終了ごとに中間モデルを保存するか(保存に時間がかかる場合はFalseにする)
saveconfig = {}
saveconfig['stage'] = 2

# 各学習ステージでのスケジュール(エポック数、カットオフエネルギー、KL学習重み)リストを定義
layer_types = 'R'*12                # セットするレイヤーの種類
nl_layers = 4                      # 隠れ層の総数 + 1
nl_hidden=[512, 254, 512]
batch_size = 1000
clipnorm = 1.0

lrs = [0.00001, 0.00005, 0.0001, 0.0001, 0.0001]
epochs_KL     = [20, 30, 30, 30, 30]
high_energies = [1e11, 1e10, 1e10, 1e9, 1e8, 1e7, 1e6, 1e5, 1e5, 1e5, 1e5, 1e4, 1e4, 1e4, 1e3, 1e3, 1e2, 1e1, 0.]
max_energies = [1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20, 1e20]
w_KLs         = [1e-13, 1e-12, 1e-12, 1e-5, 1e-4, 1e-5, 1e-5, 1e-5, 5e-5, 1e-4, 5e-4, 5e-4, 5e-3, 5e-3, 5e-2, 5e-2]
w_MLs         = [1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
w_RCs         = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
w_L2_angles   = [1e-3, 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]
rc_min=0
rc_max=53

# Data paths
# dataディレクトリに「avb3_head.pdb」を入れておく
pdb_dir = '../data/'
sim_x = np.load(pdb_dir+'sim_x.npy')

# setup ITGAVB3 energy model
def setup_AVB3(multi_gpu=False):
    """ Integrin-avb3-head エネルギーモデルをセットアップする

    Returns
    -------
    top [MDTraj Topology object]     :  AVB3のTopologyオブジェクト
    system [OpenMM System object]    :  AVB3のSystemオブジェクト
    avb3_omm_energy [Energy model]   :  AVB3のEnergy model

    """
    INTEGRATOR_ARGS = (300*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)

    from simtk.openmm import app
    # pdb構造をロードしてpdbオブジェクトを生成
    pdb = app.PDBFile(pdb_dir + 'avb3_head.pdb')

    # openMM組み込みの力場ファイルをロードしてForceFieldオブジェクトを生成 (implicit solvant[GB-obc]モデル)
    forcefield = openmm.app.ForceField('amber99sbildn.xml', 'amber99_obc.xml')
    # charmm力場を使う場合は以下をコメントアウトする
    # forcefield = openmm.app.ForceField('charmm36.xml')

    # pdbオブジェクトとForceFieldオブジェクトを合体し、計算条件を加えたsystemオブジェクトを生成
    system = forcefield.createSystem(pdb.topology, removeCMMotion=False,
                                     nonbondedMethod=app.CutoffNonPeriodic, nonbondedCutoff=1.0*unit.nanometers,
                                     constraints=None, rigidWater=True)
    # 運動方程式の積分器を定義
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
    # pdbファイルとsystemファイルと積分器をまとめて、simulationオブジェクトを生成
    simulation = openmm.app.Simulation(pdb.topology, system, integrator)
    # openMM APIを使用してエネルギー計算用のモデルを生成
    avb3_omm_energy = OpenMMEnergy(openmm_system=system,
                                   openmm_integrator=openmm.LangevinIntegrator,
                                   length_scale=unit.nanometers,
                                   n_atoms=md.Topology().from_openmm(simulation.topology).n_atoms,
                                   openmm_integrator_args=INTEGRATOR_ARGS,
                                   multi_gpu=multi_gpu)

    # MDtrajのopenMM APIを使用して、openMM用トポロジーをMDtraj用トポロジーに変換する
    mdtraj_topology = md.Topology().from_openmm(pdb.topology)
    return mdtraj_topology, system, avb3_omm_energy

# ITGAVB3モデルを定義
print('Integrin AVB3 set up\n')
sys.stdout.flush()
top, system, mm_avb3 = setup_AVB3()
print('Data loaded\n')
sys.stdout.flush()

# 軌道データのフレーム数
nframes = sim_x.shape[0]
# 各フレームの次元(3×原子数)
dim = sim_x.shape[1]

# 各原子の質量配列を取得
weights = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])
CartIndices, ZIndices = get_indices(top, cartesian_CYS=True)

if not restart:
    # ML学習済みモデルをロード
    print('Loading ML pretrained weights...'.format(saveconfig['stage']))
    sys.stdout.flush()
    s_time = time.time()
    bg = EnergyInvNet.load("./"+ weight_file_name, mm_avb3)
    load_time = time.time() - s_time
    print('Weights loaded.')
    print('Loading Time: {0}'.format(load_time) + "[sec]")
    sys.stdout.flush()

    # 反応座標関数1のセット
    ini_conf = sim_x[0]
    betaA_atoms, betaA_index = getDomainIndices(top, 'chainid 1 and not (resid 437 to 489 or resid 735 to 816)', atomwise=True)
    hybrid_atoms, hybrid_index = getDomainIndices(top, 'chainid 1 and (resid 437 to 489 or resid 735 to 816)', atomwise=True)
    weights_betaA = weights[betaA_atoms]
    weights_hybrid = weights[hybrid_atoms]
    hinge = HingeAngleRC(ini_conf=ini_conf,
                         ref_index=betaA_index, mob_index=hybrid_index,
                         ref_weights=weights_betaA, mob_weights=weights_hybrid)

    # 反応座標関数2のセット
    #_, Leu134CB = getDomainIndices(top, 'index 7799', atomwise=True)
    #_, Leu333CB = getDomainIndices(top, 'index 10863', atomwise=True)
    #LEUdist = distRC(Leu134CB, Leu333CB)

    # 2つの反応座標関数をマージ
    #MRCfunc = MergeRC(hinge, LEUdist)

else:
    print('RESTART FROM STAGE:{0}'.format(saveconfig['stage']))
    sys.stdout.flush()
    s_time = time.time()
    # リスタートする場合は以下の4行をコメントアウト (中断したステージの一つ前の重みをロード)
    print('Loading Stage{0} weights...'.format(saveconfig['stage']-1))
    sys.stdout.flush()
    bg = EnergyInvNet.load("./avb3_KL_stage{0}_saved.pkl".format(saveconfig['stage']-1), mm_avb3)
    load_time = time.time() - s_time
    print('Weights loaded.')
    print('Loading Time: {0}'.format(load_time) + "[sec]")
    sys.stdout.flush()

    # 反応座標関数1のセット
    ini_conf = sim_x[0]
    betaA_atoms, betaA_index = getDomainIndices(top, 'chainid 1 and not (resid 437 to 489 or resid 735 to 816)', atomwise=True)
    hybrid_atoms, hybrid_index = getDomainIndices(top, 'chainid 1 and (resid 437 to 489 or resid 735 to 816)', atomwise=True)
    weights_betaA = weights[betaA_atoms]
    weights_hybrid = weights[hybrid_atoms]
    hinge = HingeAngleRC(ini_conf=ini_conf,
                         ref_index=betaA_index, mob_index=hybrid_index,
                         ref_weights=weights_betaA, mob_weights=weights_hybrid)

    # 反応座標関数2のセット
    #_, Leu134CB = getDomainIndices(top, 'index 7799', atomwise=True)
    #_, Leu333CB = getDomainIndices(top, 'index 10863', atomwise=True)
    #LEUdist = distRC(Leu134CB, Leu333CB)

    # 2つの反応座標関数をマージ
    #MRCfunc = MergeRC(hinge, LEUdist)

# KL+ML+RC学習を実行する
# MLは重み付き学習ではない、またM_layerの角度損失による学習も行わない
# NOTE: 学習率(=0.001)はハードコート
print('KL Training start!!\n')
sys.stdout.flush()
train_KL(bg, sim_x, epochs_KL, high_energies, max_energies, w_KLs, lr=lrs, clipnorm=clipnorm, w_ML=w_MLs, batch_size=batch_size, stage=saveconfig['stage'],
         rc_func=hinge, rc_min=rc_min, rc_max=rc_max, multi_rc=False, w_RC=w_RCs, w_L2_angle=w_L2_angles, inter_model=inter_model)

# 学習済みモデルを保存
print('KL Training completed!!\n')
sys.stdout.flush()
s_time = time.time()
bg.save('./avb3_save_after_KL.pkl')
save_time = time.time() - s_time
print('Saving completed.\n All task finished!\n')
print('Saving Time: {0}'.format(save_time) + "[sec]")
sys.stdout.flush()
