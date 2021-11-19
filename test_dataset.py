from pie_data import PIE

data_path='../../../datasets/PIE_dataset'
dataset = PIE(data_path=data_path)
# 878 tracks
train_intention = dataset.generate_data_trajectory_sequence('train', seq_type='intention')
# 876 tracks
# train_crossing = dataset.generate_data_trajectory_sequence('train', seq_type='crossing')
# 890 tracks
# train_trajectory = dataset.generate_data_trajectory_sequence('train', seq_type='trajectory')
val_intention = dataset.generate_data_trajectory_sequence('val', seq_type='intention')
test_intention = dataset.generate_data_trajectory_sequence('test', seq_type='intention')

obs_len = 15
pred_len = 45
overlap_stride = 8
def calc_data_distribution(intent_data, obs_len, pred_len, overlap_stride):
    seq_len = obs_len + pred_len
    tracks = []
    num_positive = 0
    for track in intent_data:
        tracks.extend([track[i:i + seq_len] for i in range(0, len(track) - seq_len + 1, overlap_stride)])
        # import pdb; pdb.set_trace()
        if tracks[-1][obs_len-1][0] == 1:
            num_positive += 1
    return num_positive, len(tracks)

# import pdb;pdb.set_trace()
p, all = calc_data_distribution(test_intention['intention_binary'], obs_len, pred_len, overlap_stride)
print('test', p, all)
p, all = calc_data_distribution(val_intention['intention_binary'], obs_len, pred_len, overlap_stride)
print('val', p, all)
p, all = calc_data_distribution(train_intention['intention_binary'], obs_len, pred_len, overlap_stride)
print('train', p, all)