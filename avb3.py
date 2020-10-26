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

def get_indices(top, cartesian_CYS=True, notAminoacid=None):
    """ 各IC原子に対し二面角を構成する4原子ペアの原子番号を列にもつZ行列(2D_array)を返す
        また、内部座標に変換しないカーテシアン原子の原子番号の1D_arrayも返す
    -------------------------------------------------------------------------------------------------
    Args:
       top [MDtraj topology] : MDtrajのtopologyオブジェクト
       cartesian_CYS(=True)  : CYS残基の重原子(SG, CB)をカーテシアン原子に加えるか
                               NOTE: カーテシアン原子とは内部座標に変換しない原子のこと
       notAminoacid(=None) [list] : アミノ酸残基以外の残基がある場合は残基名(文字列)のリストをわたす

    Returns:
       cart [int_array (カーテシアン原子の原子数, )]  : カーテシアン原子の原子番号の1D_array
       Z    [int_array (二面角構成原子ペアの総数, 4)] : 内部座標に変換される原子ペアのindexの2D_array

    """
    from deep_boltzmann.models.proteins import mdtraj2Z
    cartesian = ['CA', 'C', 'N'] # backbone中の重原子(Oを除く)
    cart = top.select(' '.join(["name " + s for s in cartesian])) # CA,C,Nの原子番号の1D_ndarrayを取得
    if cartesian_CYS:
        # cartesian_CYS=Trueの場合、CYS残基のSG,CB原子を含まない4原子二面角構成ペアをZ_にリストする
        # _cartsはSG,CBの原子番号リストである
        Z_, _carts = mdtraj2Z(top,  cartesian="resname CYS and mass>2 and sidechain", notAminoacid=notAminoacid)
        Z_ = np.array(Z_) # リストのリストを2D_arrayに変換
        cart = np.sort(np.concatenate((cart,_carts))) # CA,C,Nの原子番号リストとSG,CBの原子番号リストを1D_arrayにまとめて昇順に並べる
    else:
        # cartesian_CYS=Falseの場合、CYS残基のSG,CB原子も二面角構成原子(=非カーテシアン原子)としてZ_配列にまとめる
        Z_ = np.array(mdtraj2Z(top), notAminoacid=notAminoacid)
    return cart, Z_

def getDomainIndices(top, select, atomwise=True):
    """ トポロジーオブジェクトから選択した原子やドメインのindex配列を返す
        (MDTrajトポロジー用)
        ----------------------------------------------------------------------------------
        Args:
            top [MDtrajtopology]   : MDtrajのトポロジーオブジェクト
            select [str]           : OpenMMにおけるセレクション文
                                     (indexを取得したいドメインを指定)
            atomwise(=True) [bool] : Trueにした場合、xyz座標のindex配列の形状を(-1, 3)で返す
                                     Falseの場合、xyz座標のindex配列を1D_arrayで返す
        Returns:
            select_atoms [array]        : 指定したドメインの原子番号の配列
            select_xyzindices [array]   : 指定したドメインのxyz座標の配列
    """
    dim = top.n_atoms * 3
    atom_indices = np.arange(dim).reshape((-1, 3))
    select_atoms = top.select(select)
    select_xyzindices = atom_indices[select_atoms]

    if atomwise:
        return select_atoms, select_xyzindices
    else:
        return select_atoms, select_xyzindices.flatten()


def getDomainIndicesPrody(pdb, domain, atomwise=True):
    """ トポロジーオブジェクトから選択した原子やドメインのindex配列を返す
        (Prodyトポロジー用)
        ----------------------------------------------------------------------------------
        Args:
            pdb [Prody AtomGroup]      : タンパク質全体のPrody PDBトポロジー
            domain [prody Selection]   : Prodyのセレクションオブジェクト
                                         (indexを取得したいドメイン)
            atomwise(=True) [bool]     : Trueにした場合、xyz座標のindex配列の形状を(-1, 3)で返す
                                         Falseの場合、xyz座標のindex配列を1D_arrayで返す
        Returns:
            select_xyzindices [array]  : 指定したドメインのxyz座標の配列
    """
    dim = pdb.numAtoms() * 3
    atom_indices = np.arange(dim).reshape((-1, 3))
    select_atoms = domain.getIndices()
    select_xyzindices = atom_indices[select_atoms]

    if atomwise:
        return select_xyzindices
    else:
        return select_xyzindices.flatten()

def calcCOM(conf, weights):
    """ 入力配位[conf (B,natom,3)]の重心を計算する
    """
    weights_sum = np.sum(weights)
    com = np.expand_dims(np.sum((conf * weights), axis=1) / weights_sum, axis=1)
    return com # shape (B, 1, 3)


def calcCOM_tf(conf, weights):
    """ 入力配位[conf (B,natom,3)]の重心を計算する (TensorFlow version)
    """
    weights_sum = tf.reduce_sum(weights)
    com = tf.expand_dims(tf.reduce_sum((conf * weights), axis=1) / weights_sum, axis=1)
    return com # shape (B, 1, 3)


def calcTransformMatrix(mobile, target, weights):
    """ 4×4変換(回転+並進)行列を生成する
        NOTE: mobileを動かす単一構造、targetを重ね合わせる複数構造であると仮定しているので注意
    """
    batch_size = np.shape(target)[0]
    weights_dot = np.dot(np.reshape(weights,[-1]).T, np.reshape(weights,[-1]))
    mob_com = calcCOM(mobile, weights) # shape (B, 1, 3)
    tar_com = calcCOM(target, weights) # shape (B, 1, 3)
    mobile = mobile - mob_com
    target = target - tar_com
    #tf.cast(aaa, tf.float32) matmul前にはcastする
    matrix = np.matmul(np.transpose(mobile * weights, [0,2,1]), target * weights) / weights_dot
    # 特異値分解
    U, s, Vh = np.linalg.svd(matrix)

    # 4×4Id行列の作成
    Id = np.reshape(np.tile(np.eye(2),[batch_size,1]), [batch_size,2,2])
    Id = np.concatenate([Id, np.zeros((batch_size,2,1))], axis=2) # tf.concatに変換
    det = np.reshape(np.sign(np.linalg.det(matrix)), [1,-1])
    bottom = np.transpose(np.concatenate([np.zeros([2, batch_size]), det], axis=0), [1,0])
    Id = np.concatenate([Id, np.expand_dims(bottom, axis=1)], axis=1)

    # 回転行列、並進ベクトルを計算
    rotation = np.matmul(np.transpose(Vh, [0,2,1]), np.matmul(Id, np.transpose(U, [0,2,1]))) # 回転行列
    translation = tar_com - np.matmul(mob_com, np.transpose(rotation, [0,2,1])) # 並進ベクトル
    translation = np.reshape(translation, [batch_size,3,1])

    # 4×4変換行列にまとめる
    T = np.concatenate([rotation, translation], axis=2)
    T = np.concatenate([T, np.tile(np.array([[[0.,0.,0.,1.]]]),[batch_size,1,1])], axis=1)
    return T # shape (B, 4, 4)


def calcTransformMatrix_tf(mobile, target, weights):
    """ 4×4変換(回転+並進)行列を生成する (TensorFlow version)
        NOTE: mobileを動かす単一構造、targetを重ね合わせる複数構造であると仮定しているので注意
    """
    batch_size = tf.shape(target)[0]
    weights_dot = np.dot(np.reshape(weights,[-1]).T, np.reshape(weights,[-1]))
    mob_com = calcCOM_tf(mobile, weights) # shape (B, 1, 3)
    tar_com = calcCOM_tf(target, weights) # shape (B, 1, 3)
    mobile = mobile - mob_com
    target = target - tar_com
    #tf.cast(aaa, tf.float32) matmul前にはcastする
    matrix = tf.matmul(tf.transpose(mobile * weights, [0,2,1]), target * weights) / weights_dot
    # 特異値分解
    # NOTE: TensorFlowの特異値分解はNumpyの特異値分解の出力順が違うので注意!
    #       Tf: s,U,Vh   Np: U,s,Vh
    #       また、TensorFlowのVhは転置したものがNumpyのVhに対応する
    s, U, Vh = tf.linalg.svd(matrix, full_matrices=True)
    Vh = tf.transpose(Vh, [0,2,1])

    # 4×4Id行列の作成
    Id = tf.reshape(tf.tile(tf.eye(2),[batch_size,1]), [batch_size,2,2])
    Id = tf.concat([Id, tf.zeros((batch_size,2,1))], axis=2) # tf.concatに変換
    det = tf.reshape(tf.sign(tf.linalg.det(matrix)), [1,-1])
    bottom = tf.transpose(tf.concat([tf.zeros([2, batch_size]), det], axis=0), [1,0])
    Id = tf.concat([Id, tf.expand_dims(bottom, axis=1)], axis=1)

    # 回転行列、並進ベクトルを計算
    rotation = tf.matmul(tf.transpose(Vh, [0,2,1]), tf.matmul(Id, tf.transpose(U, [0,2,1]))) # 回転行列
    translation = tar_com - tf.matmul(mob_com, tf.transpose(rotation, [0,2,1])) # 並進ベクトル
    translation = tf.reshape(translation, [batch_size,3,1])

    # 4×4変換行列にまとめる
    T = tf.concat([rotation, translation], axis=2)
    T = tf.concat([T, tf.tile(tf.constant([[[0.,0.,0.,1.]]]),[batch_size,1,1])], axis=1)
    return T # shape (B, 4, 4)


def applyTransformMatrix(T, mobile):
    """ 配位に変換行列を作用させる
    """
    rotation = T[:, :3, :3]
    translation = np.expand_dims(T[:, :3, 3], axis=1)
    new_mobile = np.matmul(mobile, np.transpose(rotation, [0,2,1])) + translation
    return new_mobile # (B, natom, 3)


def applyTransformMatrix_tf(T, mobile):
    """ 配位に変換行列を作用させる (TensorFlow version)
    """
    rotation = T[:, :3, :3]
    translation = tf.expand_dims(T[:, :3, 3], axis=1)
    new_mobile = tf.matmul(mobile, tf.transpose(rotation, [0,2,1])) + translation
    return new_mobile # (B, natom, 3)

class HingeAngleRC(object):
    """ 指定した2つのドメイン間のヒンジ角を計算するrcfunc
    """

    def __init__(self, ini_conf, ref_index, mob_index, ref_weights, mob_weights):
        """ Args:
                ini_conf [array (ndim, )]          : ヒンジ角を計算するための参照初期配位
                ref_index [array (refatoms, 3)]    : ヒンジ角を計算する際に、重ね合わせて基準とするドメインの原子座標index
                mob_index [array (mobatoms, 3)]    : ヒンジ角を計算する際に、動的に動くドメインの原子座標index
                ref_weights [array (refatoms, )]   : 重ね合わせるドメインの原子の質量配列
                mob_weights [array (refatoms, )]   : 動的なドメインの原子の質量配列
        """
        self.ini_conf = np.reshape(ini_conf, [1,-1,3]).astype(np.float32)
        self.ref_index = ref_index
        self.mob_index = mob_index
        self.ref_weights = np.expand_dims(ref_weights[:,None], axis=0).astype(np.float32)
        self.mob_weights = np.expand_dims(mob_weights[:,None], axis=0).astype(np.float32)
        self.ini_ref_conf = np.expand_dims(ini_conf, axis=0)[:, self.ref_index].astype(np.float32)


    def __call__(self, x):

        x = tf.cast(x, tf.float32)
        batch_size = tf.shape(x)[0]
        ref_conf = tf.gather(x, self.ref_index, axis=1)
        mob_conf = tf.gather(x, self.mob_index, axis=1)
        ref_T = calcTransformMatrix_tf(self.ini_ref_conf, ref_conf, self.ref_weights) # shape (B, 4, 4)

        ini_mob_conf = tf.gather(tf.reshape(applyTransformMatrix_tf(ref_T, self.ini_conf), [batch_size,-1]), self.mob_index, axis=1)
        com1 = calcCOM_tf(ini_mob_conf, self.mob_weights) # shape (B, 1, 3)
        com2 = calcCOM_tf(mob_conf, self.mob_weights) # shape (B, 1, 3)
        pl = (com2 - com1) / tf.linalg.norm(com2 - com1, axis=2, keepdims=True) # shape (B, 1, 3)

        mob_T = calcTransformMatrix_tf(ini_mob_conf, mob_conf, self.mob_weights) # shape (B, 4, 4)
        t21 = tf.reshape(tf.tile(tf.eye(3),[batch_size,1]), [batch_size,3,3])
        t21 = tf.concat([t21, tf.reshape(com1 - com2, [batch_size,3,1])], axis=2)
        t21 = tf.concat([t21, tf.tile(tf.constant([[[0.,0.,0.,1.]]]),[batch_size,1,1])], axis=1)
        rot2 = tf.matmul(mob_T, t21)

        p1 = applyTransformMatrix_tf(rot2, com1) # shape (B, 1, 3)
        p2 = applyTransformMatrix_tf(rot2, p1) # shape (B, 1, 3)
        rideal = tf.cross((com1-p2), (com1-p1))
        rideal = rideal / tf.linalg.norm(rideal, axis=2, keepdims=True)  # shape (B, 1, 3)
        new = com2 - tf.matmul(rideal, tf.transpose(com2-com1, [0,2,1])) * rideal # shape (B, 1, 3)

        cosine = tf.matmul((new-com1)/tf.linalg.norm(new-com1, axis=2, keepdims=True),
                          tf.transpose((new-p1)/tf.linalg.norm(new-p1, axis=2, keepdims=True), [0,2,1]))
        angl = tf.acos(cosine)

        perp = tf.matmul(rideal, tf.transpose(pl, [0,2,1])) # shape (B, 1, 1)
        angp = tf.abs(tf.asin(perp)) # shape (B, 1, 1)
        pro = rideal - perp * pl # shape (B, 1, 3)

        tang = tf.cos(angp) * tf.tan(0.5 * angl) # shape (B, 1, 1)
        angle = tf.reshape(2.0 * tf.atan(tang), [-1]) * 180.0 / np.pi # 度数に変換している
        return angle


    def calcRC(self, x):
        """ 与えられた軌道(バッチ)データのヒンジ角を計算する (Numpy version) """

        x = x.astype(np.float32)
        batch_size = np.shape(x)[0]
        ref_conf = x[:, self.ref_index] # tf.gatherに変える
        mob_conf = x[:, self.mob_index] # tf.gatherに変える
        ref_T = calcTransformMatrix(self.ini_ref_conf, ref_conf, self.ref_weights) # shape (B, 4, 4)

        ini_mob_conf = np.reshape(applyTransformMatrix(ref_T, self.ini_conf), [batch_size,-1])[:, self.mob_index]
        #ini_mob_conf = np.reshape(ini_mob_conf, [batch_size,-1,3])
        com1 = calcCOM(ini_mob_conf, self.mob_weights) # shape (B, 1, 3)
        com2 = calcCOM(mob_conf, self.mob_weights) # shape (B, 1, 3)
        pl = (com2 - com1) / np.linalg.norm(com2 - com1, axis=2, keepdims=True) # shape (B, 1, 3)

        mob_T = calcTransformMatrix(ini_mob_conf, mob_conf, self.mob_weights) # shape (B, 4, 4)
        t21 = np.reshape(np.tile(np.eye(3),[batch_size,1]), [batch_size,3,3])
        t21 = np.concatenate([t21, np.reshape(com1 - com2, [batch_size,3,1])], axis=2)
        t21 = np.concatenate([t21, np.tile(np.array([[[0.,0.,0.,1.]]]),[batch_size,1,1])], axis=1)
        rot2 = np.matmul(mob_T, t21)

        p1 = applyTransformMatrix(rot2, com1) # shape (B, 1, 3)
        p2 = applyTransformMatrix(rot2, p1) # shape (B, 1, 3)
        rideal = np.cross((com1-p2), (com1-p1))
        rideal = rideal / np.linalg.norm(rideal, axis=2, keepdims=True)  # shape (B, 1, 3)
        new = com2 - np.matmul(rideal, np.transpose(com2-com1, [0,2,1])) * rideal # shape (B, 1, 3)

        cosine = np.matmul((new-com1)/np.linalg.norm(new-com1, axis=2, keepdims=True),
                          np.transpose((new-p1)/np.linalg.norm(new-p1, axis=2, keepdims=True), [0,2,1]))
        angl = np.arccos(cosine) # tf.acos()に変える

        perp = np.matmul(rideal, np.transpose(pl, [0,2,1])) # shape (B, 1, 1)
        angp = np.abs(np.arcsin(perp)) # shape (B, 1, 1) tf.asin()に変える
        pro = rideal - perp * pl # shape (B, 1, 3)

        tang = np.cos(angp) * np.tan(0.5 * angl) # shape (B, 1, 1)
        angle = np.reshape(2.0*np.arctan(tang), [-1]) * 180.0 / np.pi # 度数に変換している
        return angle

class distRC(object):
    """ 2ドメイン間の平均距離(質量を与えた場合は重心間距離)を計算する
        ----------------------------------------------------------------------------------------------------
        Attributes:
            indices1 indices2 [int_array (natom, 3)]                  : ドメインのxyz座標index
            atomindices1 atomindices2 (=None) [int_array (natom, )]   : ドメインの原子index
            weights [float_array (natom, )]                           : タンパク質全体の質量

        Note:
            ドメイン間距離を計算する場合は、atomindices1とatomindices2をわたす
            weightsをわたさなかった場合は平均座標間距離になり、わたした場合は重心間距離になる
            1原子間距離の場合はatomindices1、atomindices2、weightsともにわたすす必要はない
    """

    def __init__(self, indices1, indices2, atomindices1=None, atomindices2=None,  weights=None):
        self.indices1 = indices1
        self.indices2 = indices2
        self.atomindices1 = atomindices1
        self.atomindices2 = atomindices2
        self.weights = weights
        if self.atomindices1 is not None and self.atomindices2 is not None and weights is not None:
            self.weights1 =  np.expand_dims(weights[self.atomindices1][:,None], axis=0).astype(np.float32)
            self.weights2 =  np.expand_dims(weights[self.atomindices2][:,None], axis=0).astype(np.float32)


    def __call__(self, x):
        if tf.shape(self.indices1)[0] != 1 and tf.shape(self.indices2)[0] != 1:
            if self.weights is not None:
                x = tf.cast(x, tf.float32)
                com1 = calcCOM_tf(tf.gather(x, self.indices1, axis=1), self.weights1)
                com2 = calcCOM_tf(tf.gather(x, self.indices2, axis=1) , self.weights2)
                return tf.reshape(tf.linalg.norm(com2 - com1, axis=2, keepdims=True), [-1])
            else:
                mean1 = tf.reduce_mean(tf.gather(x, self.indices1, axis=1) , axis=1)
                mean2 = tf.reduce_mean(tf.gather(x, self.indices2, axis=1) , axis=1)
                return tf.linalg.norm(mean2 - mean1, axis=1)
        else:
            return tf.reshape(tf.linalg.norm(tf.gather(x, self.indices2, axis=1) - tf.gather(x, self.indices1, axis=1), axis=2), [-1])


    def calcRC(self, x):
        if np.shape(self.indices1)[0] != 1 and np.shape(self.indices2)[0] != 1:
            if self.weights is not None:
                x = x.astype(np.float32)
                com1 = calcCOM(x[:, self.indices1], self.weights1)
                com2 = calcCOM(x[:, self.indices2], self.weights2)
                return np.reshape(np.linalg.norm(com2 - com1, axis=2, keepdims=True), [-1])
            else:
                mean1 = np.mean(x[:, self.indices1], axis=1)
                mean2 = np.mean(x[:, self.indices2], axis=1)
                return np.linalg.norm(mean2 - mean1, axis=1)
        else:
            return np.reshape(np.linalg.norm(x[:,self.indices2] - x[:,self.indices1], axis=2), [-1])

class MergeRC(object):
    """ 2つの1-output rcfuncの出力結果をマージして2-output rcfuncに変換する
    ----------------------------------------------------------------------------
    Attributes:
        rcfunc1, rcfunc2 [rcfunction] : 1D-arrayで返す1-output rcfuncオブジェクト
        outputdims(=2) [int]          : 1にすると1D_arrayで結果を返す

    """

    def __init__(self, rcfunc1, rcfunc2, outputdim=2):
        self.rcfunc1 = rcfunc1
        self.rcfunc2 = rcfunc2
        self.outputdim = outputdim


    def __call__(self, x):
        rc1 = tf.expand_dims(self.rcfunc1(x), axis=1)
        rc2 = tf.expand_dims(self.rcfunc2(x), axis=1)
        rc12 = tf.concat([rc1, rc2], axis=1)
        if self.outputdim == 1:
            return tf.reshape(rc12, [-1]) # shape (2B, )
        else:
            return rc12 # shape (B, 2)


    def calcRC(self, x):
        rc1 = np.expand_dims(self.rcfunc1.calcRC(x), axis=1)
        rc2 = np.expand_dims(self.rcfunc2.calcRC(x), axis=1)
        rc12 = np.concatenate([rc1, rc2], axis=1)
        if self.outputdim == 1:
            return np.reshape(rc12, [-1]) # shape (2B, )
        else:
            return rc12 # shape (B, 2)

def train_ML(bg, xtrain, epochs, batch_sizes, lr=0.001, clipnorm=None, counter=0, log_file='train_ML.log', file_path='./', log_stride=1):
    """ 入力BGモデルに対し、バッチサイズを変えるスケジューリングML学習を実行する
        -------------------------------------------------------------------------------
        Args:
            bg [model]                            : 学習するBoltzmann Generator
            xtrain [float_array (nsample, ndim)]  : ML学習に用いる軌道データ
            epochs [int]                          : 学習するエポック数
            batch_sizes [int_list]                : 各スケジュールでのバッチ数のリスト

        Note:
            総学習イテレーション数は "batch_sizesリストの要素数 × epochs" で決まる

    """
    trainer_ML = MLTrainer(bg, lr=lr, clipnorm=clipnorm)
    with open(log_file, "w") as f:
        for i, batch_size in enumerate(batch_sizes):
            start_time = time.time()
            trainer_ML.train(xtrain, epochs=epochs, batch_size=batch_size, log_file=f, log_stride=log_stride)
            stage_time = time.time() - start_time
            print('Time spent at Stage{0}:{1}'.format(counter, stage_time) + "[sec]", file=f)

            save_start = time.time()
            print('Intermediate model is now saving...', file=f)
            bg.save(file_path + 'avb3_ML_stage{0}_saved.pkl'.format(counter))
            print('Intermediate result saved', file=f)
            save_time = time.time() - save_start
            print('Model Saving Time at Stage{0}:{1}'.format(counter, save_time) + "[sec]", file=f)
            sys.stdout.flush()
            counter += 1


def train_KL(bg, xtrain, epochs, high_energies, max_energies, w_KLs, lr=0.0001, clipnorm=1.0, batch_size=128, w_ML=1., weigh_ML=False, stage=0, rc_func=None, rc_min=-1,
             rc_max=6, multi_rc=False, w_RC=1., w_L2_angle=0., file_path='./', log_file=None, log_stride=1, inter_model=True):
    """ 入力BGモデルに対し、エポック数、カットオフエネルギー、KL重みを変えるスケジューリングKL+ML+RC学習を実行する
        スケジュールの各ステージごとに途中経過として学習済み重みデータをpickle形式で保存する
        NOTE: スケジュールごとにバッチサイズ(=5000)、学習率(=0.001)は固定であるとする
        ----------------------------------------------------------------------------------------------------------------------------
        Args:
            bg [model]                               : 学習するBoltzmann Generator
            xtrain [float_array (nsample, ndim)]     : ML学習に用いる軌道データ
            epochs [int_list]                        : スケジュールのステージごとのエポック数リスト
            high_energies [float_list]               : スケジュールのステージごとのカットオフエネルギーリスト
            w_KLs [float_list]                       : スケジュールのステージごとのKL学習重みリスト
            batch_size(=128) [int]                   : 各ステージのバッチサイズ
                                                       NOTE: メモリオーバーになったときはbatch_sizeを少なくして
                                                             stage数を増やす
            w_ML(=1.) [float]                        : ML学習の重み
            weigh_ML(=False) [bool]                  : 重み付きML学習にするかどうか
            stage(=0) [int]                          : スケジュールのどのステージから学習をはじめるか
                                                       NOTE: 何らかの問題で学習が中断した時は、中断したステージ番号を
                                                             この引数にわたすことで、学習をリスタートできる
            rc_func(=None) [function]                : Tensorflowで書かれた反応座標関数
            rc_min(=-1), rc_max(=6) [float or list]  : 反応座標の探索範囲の最小値/最大値
                                                       NOTE: rcfuncを複数output(複数反応座標リスト)で設計した場合
                                                             rc_min=[反応座標1の最小値, ..., 反応座標Nの最小値]
                                                             rc_max=[反応座標1の最大値, ..., 反応座標Nの最大値]
                                                             のようなリストでわたす
            multi_rc(=False) [bool]                  : rcfuncが複数output(複数反応座標リスト)で設計した場合Trueにする
            w_RC(=1.) [float]                        : RC学習の重み
            w_L2_angle(=0.) [float]                  : M_layerの角度損失による学習の重み
            file_path(='./') [str]                   : 中間学習重みデータを保存するディレクトリパス
            log_file(=None) [str]                    : 損失関数の値を出力するlogファイルのパスと名前 NOTE: Noneの場合、ステージごとに分けてログが出力される
            log_stride(=1) [int]                     : logファイルに出力するエポックの間隔
            inter_model(=True)                       : 中間ステージでの学習モデルを保存するか(保存に時間がかかる場合はFalseにする)

        Note:
           各学習スケジュールのステージごとに、学習したモデルで10,000配位を生成しそのエネルギー分布を調べる
           各ステージのカットオフエネルギー(high_energies)以上の生成配位数を標準出力する
           また、各ステージで学習に要した時間も秒単位で出力する

    """
    trainers_KL_state3 = []
    counter = 0

    for current_stage in range(stage, len(epochs)):
        if log_file is None:
            log_file = f'train_KL_stage{current_stage}.log'
        with open(log_file, "w") as f:
            # スケジュールの各ステージごとに学習イテレーション
            if counter == 0: # 開始ステージ(リスタートなら途中から)の時刻とステージ番号を保存
                root_time = time.time()
                root_stage = current_stage

            # 現在のイテレーションの開始時刻を保存
            start_time = time.time()

            # 現在のステージの情報を標準出力
            print('-----------------------', file=f)
            print('Stage:{0}\n high_energy={1}  w_KL={2}'.format(current_stage, high_energies[current_stage], w_KLs[current_stage]), file=f)
            sys.stdout.flush()

            # 学習スケジュールに則りトレイナーを定義し、学習を実施
            flextrainer = FlexibleTrainer(bg, lr=lr[current_stage], clipnorm=clipnorm, batch_size=batch_size, high_energy=high_energies[current_stage], max_energy=max_energies[current_stage],
                                          w_KL=w_KLs[current_stage], w_ML=w_ML[current_stage], weigh_ML=weigh_ML, w_RC=w_RC[current_stage],
                                          rc_func=rc_func, rc_min=np.array(rc_min), rc_max=np.array(rc_max), multi_rc=multi_rc,
                                          w_L2_angle=w_L2_angle[current_stage])
            flextrainer.train(xtrain, epochs=epochs[current_stage], log_file=f, log_stride=log_stride)
            trainers_KL_state3.append(flextrainer)

            # 学習後の中間BGモデルで配位を生成し、エネルギーを計算する
            samples_z = np.random.randn(10000, bg.dim)
            samples_x = bg.Tzx.predict(samples_z)
            samples_e = bg.energy_model.energy(samples_x)
            # 生成配位のエネルギーの内、カットオフエネルギーより大きなエネルギーの配位の総数をステージごとに計算してリストにする
            energy_violations = [np.count_nonzero(samples_e > E) for E in high_energies]
            # 生成配位のエネルギーについての情報を標準出力する
            print('Energy violations: Total number of generated samples with energies higher than high_energies', file=f)
            for i, (E, V) in enumerate(zip(high_energies, energy_violations)):
                print('NUM of samples:', V, '\t>\t', 'high_energy at Stage{0}:'.format(i), E, file=f)
            sys.stdout.flush()

            # 学習に要した時間を標準出力する
            stage_time = time.time() - start_time
            total_time = time.time() - root_time
            print('Time spent at Stage{0}:{1}'.format(current_stage, stage_time) + "[sec]", file=f)
            print('Total time from Stage{0}:{1}'.format(root_stage, total_time) + "[sec]", file=f)
            sys.stdout.flush()

            # 現在のステージで学習した中間BGモデルをpickleで保存 (保存に時間がかかる場合がある)
            if inter_model:
                save_start = time.time()
                print('Intermediate model is now saving...', file=f)
                bg.save(file_path + 'avb3_intermediate_model_stage{0}_saved.pkl'.format(current_stage))
                print('Intermediate result saved', file=f)
                save_time = time.time() - save_start
                print('Model Saving Time at Stage{0}:{1}'.format(current_stage, save_time) + "[sec]", file=f)
                sys.stdout.flush()

            counter += 1
