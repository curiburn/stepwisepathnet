import tensorflow as tf
from tensorflow import keras
import copy
import numpy as np
import sys

from scipy.stats import truncnorm 
from numpy.random import choice 


class sw_pathnet:
    # コンストラクタ
    def __init__(self, pre_model, n_comp, n_classes, transfer_all_layer, is_reuse_initweight=False):
        print(pre_model.summary())
        self.n_classes = n_classes
        self.transfer_all_layer = transfer_all_layer
        
        # 学習用のテンプレとして，pre_modelを複製
        #   VRAM上のモデル単位のメモリ解放ができない臭い
        self.tmp_model = self.gen_tmp_model(pre_model)
        
        # 重みのある層の取得
        #   batchnormのせいで4も省かないと<-BN層も入れないとまずいんじゃないか
        #self.li_is_weighted = self.gen_li_weighted(self.tmp_model)
        self.li_is_weighted = [self.is_weighted(l) for l in self.tmp_model.layers]
        print(self.li_is_weighted)
        
        # パラメータの保存
        #   学習済みモデルに合わせて作成，重みがある箇所は初期値を置く
        self.source_weights = []
        self.target_weights = []
        self.len_geopath = 0
        self.initializer_k = truncnorm(-1, 1, loc=0, scale=0.25)
        self.val_b = 0.0
        for i_layer in range(len(self.tmp_model.layers)):
            # 重みのある層のみの処理
            if self.li_is_weighted[i_layer]:
                # len_geopathの更新
                self.len_geopath += 1
                
                # ソースの重みの格納
                self.source_weights.append(self.tmp_model.layers[i_layer].get_weights())
                
                # ターゲットの重みの初期値の格納
                #   学習済みの重みを使うかで分岐
                #   最終層のみは学習済みの重みを使うかに依らずに生成
                #   kernel, biasの順（変わらないでくれ頼む）
                if is_reuse_initweight and i_layer != len(self.tmp_model.layers)-1:
                    # 学習済みの重みを引っ張るだけ
                    self.target_weights.append(self.tmp_model.layers[i_layer].get_weights())
                    
                else:
                    print('layer: ', i_layer)
                    
                    tmp_target_weight = []
                    for weight in self.tmp_model.layers[i_layer].weights:
                        print('given', weight.shape)
                        # shapeのlenでbiasかkernelかを判定
                        tmp_target_weight.append(self.get_init_weight(weight.shape))
                        #if len(weight.shape) == 1:
                        #    b_len = weight.shape[0]
                        #    tmp_target_weight.append(
                        #        np.array([val_b for i in range(b_len)]))
                        #    
                        #else:
                        #    tmp_target_weight.append(self.get
                        #        initializer_k.rvs(weight.shape))
                        print('generated', tmp_target_weight[-1].shape)
                    
                    self.target_weights.append(tmp_target_weight)
            else:
                # 重みなしレイヤーは空リストを追加
                self.source_weights.append([])
                self.target_weights.append([])
        
        
    # 重みの初期値の生成
    #   1次元ならバイアス，2次元なら畳み込みor全結合層　
    def get_init_weight(self, weight_shape):
        if len(weight_shape) == 1:
            init_weight = np.array([self.val_b for i in range(weight_shape[0])])
        else:
            init_weight = self.initializer_k.rvs(weight_shape)
        return init_weight


    # 重みのあるレイヤーがTrueになるリストを返す
    def is_weighted(self, layer):
        if self.transfer_all_layer:
            not_weighted = len(layer.weights) == 0
        else:
            not_weighted = len(layer.weights) == 0 or len(layer.weights) == 4 or len(layer.weights) == 3
        
        return not(not_weighted)


    def gen_li_weighted(self, model):
        li_is_weighted = []
        for i_layer in range(len(model.layers)):
            if self.is_weighted(model.layers[i_layer]):
                li_is_weighted.append(False)
            else:
                li_is_weighted.append(True)
        
        return li_is_weighted
    
    
    # top-layerだけn_classesに合わせたモデルを生成
    #   https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model より　笹川実装を参考　
    def gen_tmp_model(self, pre_model):
        tmp_li_is_weighted = [self.is_weighted(l) for l in pre_model.layers]
        i_last_weighted = -1 - tmp_li_is_weighted[::-1].index(True)
        print(i_last_weighted, 'will be replaced to dense with', self.n_classes)
        
        tmp_li_is_weighted = tmp_li_is_weighted[:i_last_weighted]
        i_replaced = -1 - tmp_li_is_weighted[::-1].index(True)
        
        x = pre_model.layers[i_last_weighted-1].output
        predictions = keras.layers.Dense(self.n_classes, activation='softmax', name='dense_top')(x)
        tmp_model = keras.models.Model(inputs=pre_model.input, outputs=predictions)
        
        print(tmp_model.summary())
        
        return tmp_model
        
        
    # 遺伝子型の生成
    #   デフォはソースタスクレイヤーと新規学習レイヤーが当確率で選ばれる
    def gen_geopath(self, bias_pretrained=0.5, is_top_unfixed=True):
        geopath = choice(
            [0, 1], self.len_geopath, 
            p=[1-bias_pretrained, bias_pretrained])
        
        if is_top_unfixed:
            geopath[-1] = 1
        
        return geopath
    
    
    # 遺伝子型からmodelへの変換
    #   0: 学習済みモデルのモジュール
    #   1: 新しく学習するモジュール
    def gene2model(self, gene):
        # copy pretrained model
        model = self.tmp_model
        model.reset_states()
        
        i_gene = 0
        for i_layer in range(len(model.layers)):
            # 重みがあればgeneを参照してtrainableを変更
            #   新規学習レイヤーの場合は重みをロード
            if self.li_is_weighted[i_layer]:
                if gene[i_gene] == 0:
                    model.layers[i_layer].trainable=False
                    model.layers[i_layer].set_weights(self.source_weights[i_layer])
                elif gene[i_gene] == 1:
                    model.layers[i_layer].trainable=True
                    model.layers[i_layer].set_weights(self.target_weights[i_layer])
                else:
                    sys.exit('invalid gene value %s' % gene)
                
                # geneのイテレータを増やす
                i_gene += 1
        
        return model
                    
    
    # 突然変異
    #   最終識別層以外を1/(層の数+1)の確率で反転
    def mutate_geopath(self, geopath):
        # 突然変異させる箇所を乱数で生成
        i_rand = int(np.random.rand() * (self.len_geopath))
        
        # i_randがout-of-rangeでなければ突然変異
        if i_rand < self.len_geopath - 1:
            is_mutated = True
            if geopath[i_rand] == 1:
                geopath[i_rand] = 0
            elif geopath[i_rand] == 0:
                geopath[i_rand] = 1
            else:
                sys.exit('invalid geopath value %s' % geopath)
            
        return geopath
    
    
    # 新規学習レイヤーの重みを保存
    def store_weights(self, gene, weights):
        i_gene = 0
        for i_layer in range(len(self.li_is_weighted)):
            if self.li_is_weighted[i_layer]:
                if gene[i_gene] == 1 and i_gene != 0:
                    self.target_weights[i_layer] = weights[i_layer]
                    
                i_gene += 1
    
    
    # kerasなmodelからの重みの抽出
    def extract_weights(self, model):
        tmp_weights = []
        for i_layer, is_weighted in enumerate(self.li_is_weighted):
            if is_weighted:
                tmp_weights.append(model.layers[i_layer].get_weights())
            else:
                tmp_weights.append([])
        return tmp_weights
