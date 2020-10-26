import matplotlib.pyplot as plt
import numpy as np
import mdtraj
from deep_boltzmann.networks.training import MLTrainer, FlexibleTrainer
from deep_boltzmann.networks.invertible import invnet, EnergyInvNet
from deep_boltzmann.models.openmm import OpenMMEnergy
import mdtraj as md
from simtk import openmm, unit
import sys, os, shutil
import time
import tensorflow as tf
from avb3 import *

# ハイパーパラメータ
layer_types = 'R'*12               # セットするレイヤーの種類
nl_layers = 4                      # 隠れ層の総数 + 1
nl_hidden=[512, 254, 512]          # 隠れ層のノード数
epochs_ML = 1000                   # エポック数
batch_sizes_ML = [512]             # バッチサイズ
current_stage = 5                  # 現在の学習stage
restart = True                     # リスタートするか
weight_file_name = "avb3_ML_stage4_saved.pkl" # restart=Trueのとき、保存した重みデータ名を与える
lr = 0.0005                         # 学習率
clipnorm = 0.6                     # 勾配クリッピング

# Data paths
# dataディレクトリに「avb3_head.pdb」と「sim_x.npy」を入れておく
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

# MD軌道の座標配列データを読み込む
ini_conf = sim_x[0]
#np.save('ini_conf', ini_conf)

# 軌道データのフレーム数
nframes = sim_x.shape[0]
# 各フレームの次元(3×原子数)
dim = sim_x.shape[1]

# フレーム同士をシャッフル
#np.random.shuffle(sim_x)
# 訓練サンプルを保存
#np.save('sim_x', sim_x)

print('Data loaded\n')
sys.stdout.flush()

# 各原子の質量配列を取得
weights = np.array([system.getParticleMass(i).value_in_unit(unit.dalton) for i in range(system.getNumParticles())])
# 主鎖原子の原子番号indexと側鎖原子のZ matrixを取得
CartIndices, ZIndices = get_indices(top, cartesian_CYS=True)

# BGを定義
# RealNVPを8層を使用し,RealNVPにセットする全結合層は3つの隠れ層と1つの出力層をもつ
# 各隠れ層のノード数を個別に指定するため、nl_hidden引数をリストでわたしていることに注意
print('BG now set up...\n')

start_time = time.time()

if not restart:
    bg = invnet(dim, layer_types, energy_model=mm_avb3,
                ic=ZIndices, ic_cart=CartIndices, ic_norm=sim_x,
                nl_layers=nl_layers, nl_hidden=nl_hidden, nl_activation='relu', nl_activation_scale='tanh')
else:
    bg = EnergyInvNet.load("./"+ weight_file_name, mm_avb3)

load_time = time.time() - start_time
print('BG constructed\n')
print('Time spent at loading:{0}'.format(load_time) + "[sec]")
sys.stdout.flush()

# バッチサイズを変更させてML学習を3回(×2000エポック)行う
print('ML Training Start !!\n')
sys.stdout.flush()
train_ML(bg, sim_x, epochs_ML, batch_sizes_ML, lr=lr, clipnorm=clipnorm, counter=current_stage)
print('ML Training Finished !!\n')
sys.stdout.flush()

# 学習済みモデルを保存
#print('ML Training completed!!\n Trained model is now saving...\n')
#sys.stdout.flush()
#s_time = time.time()
#bg.save('./avb3_save_after_ML.pkl')
#save_time = time.time() - s_time
#print('Saving completed.\n All task finished!\n')
#print('Saving Time: {0}'.format(save_time) + "[sec]")
#sys.stdout.flush()

samples_z = np.random.randn(10000, bg.dim)
samples_x = bg.Tzx.predict(samples_z)
samples_e = bg.energy_model.energy(samples_x)

high_energies = [1e26, 1e25, 1e24, 1e23, 1e22, 1e21, 1e20, 1e19, 1e18, 1e17, 1e16, 1e15, 1e14, 1e13, 1e12, 1e11,
                 1e10, 1e9, 1e8, 1e7, 1e6, 1e5, 1e4, 1e3]

# 生成配位のエネルギーの内、カットオフエネルギーより大きなエネルギーの配位の総数をステージごとに計算してリストにする
energy_violations = [np.count_nonzero(samples_e > E) for E in high_energies]
# 生成配位のエネルギーについての情報を標準出力する
print('Energy violations: Total number of generated samples with energies higher than high_energies')
for i, (E, V) in enumerate(zip(high_energies, energy_violations)):
    print('NUM of samples:', V, '\t>\t', 'high_energy at Stage{0}:'.format(i), E)
sys.stdout.flush()
