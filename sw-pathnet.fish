# CIFAR-100
for i in (seq 0 9)
    python3 sw-pathnet-orig_tournament.py $i ./cifar/cifar100 ./ 100 50000 10000  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name inceptionv3 --n_thread 8 2>&1 | tee inceptionv3_orig-tournamet_$i.log
end
for i in (seq 0 9)
    python3 sw-pathnet-mod_tournament.py $i ./cifar/cifar100 ./ 100 50000 10000  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name inceptionv3 --n_thread 8 2>&1 | tee inceptionv3_mod-tournamet_$i.log
end
for i in (seq 0 9)
    python3 sw-pathnet-mod_tournament.py $i ./cifar/cifar100 ./ 100 50000 10000  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name inceptionv3 --n_thread 8 --finetune 2>&1 | tee inceptionv3_mod-tournamet_$i.log
end

# SVHN
for i in (seq 0 9)
    python3 sw-pathnet-orig_tournament.py $i ./svhn ./ 100 73257 26032  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name inceptionv3 --n_thread 8 2>&1 | tee inceptionv3_orig-tournamet_$i.log
end
for i in (seq 0 9)
    python3 sw-pathnet-mod_tournament.py $i ./svhn ./ 100 73257 26032  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name inceptionv3 --n_thread 8 2>&1 | tee inceptionv3_mod-tournamet_$i.log
end
for i in (seq 0 9)
    python3 sw-pathnet-mod_tournament.py $i ./svhn ./ 100 73257 26032  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name inceptionv3 --n_thread 8 --finetune 2>&1 | tee inceptionv3_mod-tournamet_$i.log
end

# Food-101
for i in (seq 0 9)
    python3 sw-pathnet-orig_tournament.py $i ./svhn ./ 100 75750 25250  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name inceptionv3 --n_thread 8 2>&1 | tee inceptionv3_orig-tournamet_$i.log
end
for i in (seq 0 9)
    python3 sw-pathnet-mod_tournament.py $i ./svhn ./ 100 75750 25250  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name inceptionv3 --n_thread 8 2>&1 | tee inceptionv3_mod-tournamet_$i.log
end
for i in (seq 0 9)
    python3 sw-pathnet-mod_tournament.py $i ./svhn ./ 100 75750 25250  --image_size 224 --batch_size 16 --epochs 30 --use_augument --model_name inceptionv3 --n_thread 8 --finetune 2>&1 | tee inceptionv3_mod-tournamet_$i.log
end
