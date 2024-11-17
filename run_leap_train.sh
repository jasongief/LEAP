loss_temporal_wei=1
loss_avss_wei=1
eps=0.01
avl_head=8
python main_leap.py \
--mode 'train' \
--v_pseudo_flag \
--a_pseudo_flag \
--temporal_a_loss_flag \
--temporal_v_loss_flag \
--loss_temporal_wei ${loss_temporal_wei} \
--avss_loss_flag \
--loss_avss_wei ${loss_avss_wei} \
--use_pseudo_label \
--eps ${eps} \
--lv_layer_num 2 \
--la_layer_num 2 \
--avl_head ${avl_head} \
--lr 1e-4 \
--batch_size 32 \
--epochs 30 \
--model "MM_Pyramid_temporal_Lav_Tav_way0_lr1e-4_bs32_layers2_protoloss3_mse1" \
--model_save_dir "./pretrained_model" \