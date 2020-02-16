# CIFAR100
for i in (seq 0 9); python3 scratch.py $i ./cifar/cifar100 ./result_scratch --num_classes 100  --model_name inceptionv3 --n_thread 8  --num_images_train 50000 --num_images_test 10000 2>&1 | tee result_scratch/$i.log; end
for i in (seq 0 9); python3 finetuning.py $i ./cifar/cifar100 ./result_finetuning --num_classes 100 --model_name inceptionv3 --n_thread 8  --num_images_train 50000 --num_images_test 10000 2>&1 | tee result_finetuning/$i.log; end

# SVHN
for i in (seq 0 9); python3 scratch.py $i ./svhn ./result_scratch --num_classes 10  --model_name inceptionv3 --n_thread 8  --num_images_train 73257 --num_images_test 26032 2>&1 | tee result_scratch/$i.log; end
for i in (seq 0 9); python3 finetuning.py $i ./svhn ./result_finetuning --num_classes 10 --model_name inceptionv3 --n_thread 8  --num_images_train 73257 --num_images_test 26032 2>&1 | tee result_finetuning/$i.log; end

# Food-101
for i in (seq 0 9); python3 scratch.py $i ./food-101 ./result_scratch --num_classes 101  --model_name inceptionv3 --n_thread 8  --num_images_train 75750 --num_images_test 25250 2>&1 | tee result_scratch/$i.log; end
for i in (seq 0 9); python3 finetuning.py $i ./food-101 ./result_finetuning --num_classes 101 --model_name inceptionv3 --n_thread 8  --num_images_train 75750 --num_images_test 25250 2>&1 | tee result_finetuning/$i.log; end
