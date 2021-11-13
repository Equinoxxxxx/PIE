from pie_data import PIE

data_path='../../../datasets/PIE_dataset'
dataset = PIE(data_path=data_path)
# 878 tracks
train_intention = dataset.generate_data_trajectory_sequence('train', seq_type='intention')
# 876 tracks
train_crossing = dataset.generate_data_trajectory_sequence('train', seq_type='crossing')
# 890 tracks
train_trajectory = dataset.generate_data_trajectory_sequence('train', seq_type='trajectory')
val_intention = dataset.generate_data_trajectory_sequence('val', seq_type='intention')
import pdb;pdb.set_trace()