import torch
from torchvision import transforms
from torchvision.datasets import UCF101

ucf_data_dir= ""
ucf_data_dir= ""

frames_per_clip=5
step_between_clips=1
batch_size=32

tfs= transforms.Compose([
    transforms.Lambda(lambda x:x/255.),
    transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
    transforms.Lambda(lambda x: nn.functional.interpolate(x, (240, 320))),
])

def custom_collate(batch):
    filtered_batch=[]
    for video, _, label in batch:
        filtered_batch.append((video, label))
    return torch.utils.data.dataloader.default_collate(filtered_batch)

train_dataset= UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip, 
               step_between_clips=step_between_clips, train=True, transform=tfs)

train_loader= torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

test_dataset= UCF101(ucf_data_dir, ucf_label_dir, frames_per_clip=frames_per_clip, step_between_clips=step_between_clips, train=False, transform=tfs)

test_loader= torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)


