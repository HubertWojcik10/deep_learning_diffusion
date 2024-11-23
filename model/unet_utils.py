import torch.nn as nn
from typing import Tuple, Union, List

class UnetUtils:
    """
        Util functions for the Unet architecture.
    """
    @staticmethod
    def temb_layer(time_emb_dim: int, in_channels: int, out_channels:int, block: str = "down") -> Tuple[nn.Sequential, nn.Sequential]:
        """
            Initialize the time embedding layer.
            The same structure for each block.
        """
        time_emb_layer1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels if block != "up" else in_channels)
        )
        time_emb_layer2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        return time_emb_layer1, time_emb_layer2
    
    @staticmethod
    def resnet1(in_channels: int, out_channels: int, dropout_rate: float=0., group_num: int=8, block: str = "down") -> Tuple[nn.Sequential, nn.Sequential]:
        """
            Initialize the first resnet.
            Different structure for each block.
        """
        # down and mid have the same structure apart from number of groups (GroupNorm)
        if block == "down" or block == "mid":
            resnet_conv1 = nn.Sequential(
                nn.GroupNorm(1 if block == "down" else group_num, in_channels),
                nn.SiLU(),
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            resnet_conv2 = nn.Sequential(
                nn.GroupNorm(group_num, out_channels),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            return resnet_conv1, resnet_conv2
        elif block == "up":
            resnet_conv1 = nn.Sequential(
                nn.GroupNorm(group_num, in_channels*2),
                nn.SiLU(),
                nn.Conv2d(in_channels*2, in_channels, kernel_size=3, stride=1, padding=1)
            )
            resnet_conv2 = nn.Sequential(
                nn.GroupNorm(group_num, in_channels),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            return resnet_conv1, resnet_conv2
        else:
            raise ValueError("Block has to be equal to down, mid, or up.")
        
    @staticmethod
    def resnet2(in_channels: int, out_channels: int, group_num: int=8, dropout_rate: float=0., block:str ="down") -> Tuple[nn.Sequential, nn.Sequential]:
        """
            Initialize the second resnet.
            Different structure for each block.
        """
        if block == "down" or block == "mid":
            resnet_conv3 = nn.Sequential(
                nn.GroupNorm(group_num, out_channels),
                nn.SiLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            resnet_conv4 = nn.Sequential(
                nn.GroupNorm(group_num, out_channels),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            return resnet_conv3, resnet_conv4
        elif block == "up":
            resnet_conv3 = nn.Sequential(
                nn.GroupNorm(group_num, in_channels*2),
                nn.SiLU(),
                nn.Conv2d(in_channels*2, out_channels, kernel_size=3, stride=1, padding=1)
            )
            resnet_conv4 = nn.Sequential(
                nn.GroupNorm(group_num, out_channels),
                nn.SiLU(),
                nn.Dropout(dropout_rate),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            )
            return resnet_conv3, resnet_conv4
        else: 
            raise ValueError("Block has to be equal to down, mid, or up.")
    
    @staticmethod
    def attentions(in_channels: int, out_channels: int, num_groups: int = 8, att_head_num: int = 4, block: str = "down") -> Union[
        Tuple[nn.GroupNorm, nn.MultiheadAttention],
        Tuple[nn.GroupNorm, nn.MultiheadAttention, nn.GroupNorm, nn.MultiheadAttention]
    ]:
        """
            Initialize attentin norms and multihead attention.
            Different structure for each block. 
        """
        if block == "mid":
            attention_norms = nn.GroupNorm(num_groups, out_channels)
            attentions = nn.MultiheadAttention(out_channels, att_head_num, batch_first=True)

            return attention_norms, attentions 
        elif block == "down" or block == "up":
            attention_norms1 = nn.GroupNorm(num_groups, out_channels if block != "up" else in_channels)
            attentions1 = nn.MultiheadAttention(out_channels if block != "up" else in_channels, att_head_num, batch_first=True)

            attention_norms2 = nn.GroupNorm(num_groups, out_channels)
            attentions2 = nn.MultiheadAttention(out_channels, att_head_num, batch_first=True)

            return attention_norms1, attentions1, attention_norms2, attentions2
        else:
            raise ValueError("Block has to be equal to down, mid, or up.")

    @staticmethod
    def down_sample(out_channels:int) -> nn.Conv2d:
        """
            Create conv2d used for downsampling in the down block.
        """
        return nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    @staticmethod
    def up_sample(in_channels:int) -> nn.ConvTranspose2d:
        """
            Create conv2dTranspose2d used for upsampling in the up block.
        """
        return nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
    
    @staticmethod
    def time_projection(time_emb_dim: int) -> nn.Sequential:
        """
            Create Unet's time projections.
        """
        return nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
    
    @staticmethod
    def conv_in(im_channels: int, channels_lst: List[int]) -> nn.Conv2d:
        """ 
            Create the initial convolutional layer.
        """
        return nn.Conv2d(im_channels, channels_lst[0], kernel_size=3, padding=(1, 1))
    
    @staticmethod
    def conv_out(im_channels:int, channels_lst: List[int], group_num: int=8) -> Tuple[nn.GroupNorm, nn.Conv2d]:
        """
            Create the convolutional out layer.
        """
        norm_out = nn.GroupNorm(group_num, channels_lst[0])
        conv_out = nn.Conv2d(channels_lst[0], im_channels, kernel_size=3, padding=(1, 1))

        return norm_out, conv_out
