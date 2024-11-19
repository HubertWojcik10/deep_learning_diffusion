import torch
import torch.nn as nn
from utils.sinusoidal_embeddings import get_sinusoidal_embeddings

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, att_head_num=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.att_head_num = att_head_num

        self._init_resnet1()
        self._init_time_emb_layer()
        self._init_resnet2()
        self._init_attentions()
        #self._init_residual_input_conv()
        self._init_downsample()


    def _init_resnet1(self, group_num=8, dropout_rate=0.):
        self.resnet_conv1 = nn.Sequential(
            nn.GroupNorm(1, self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.resnet_conv2 = nn.Sequential(
            nn.GroupNorm(group_num, self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate), # fix: dropout rate is 0.
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
    
    def _init_time_emb_layer(self):
        """ Initialize a layer for sinusoidal time embedings """ 
        self.time_emb_layer1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.out_channels)
        )
        self.time_emb_layer2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.out_channels)
        )
    
    def _init_resnet2(self, group_num=8, dropout_rate=0.): 
        """ Initialize the second resnet, NOTE: find out whether dropout should be used """
        self.resnet_conv3 = nn.Sequential(
            nn.GroupNorm(group_num, self.out_channels),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.resnet_conv4 = nn.Sequential(
            nn.GroupNorm(group_num, self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate), # fix: dropout rate is 0.
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
    
    
    def _init_attentions(self, num_groups=8):
        """ GroupNorm, transpose, and initialize the attention """
        self.attention_norms1 = nn.GroupNorm(num_groups, self.out_channels)
        self.attentions1 = nn.MultiheadAttention(self.out_channels, self.att_head_num, batch_first=True)

        self.attention_norms2 = nn.GroupNorm(num_groups, self.out_channels)
        self.attentions2 = nn.MultiheadAttention(self.out_channels, self.att_head_num, batch_first=True)
    
    def _init_residual_input_conv(self):
        """ NOTE: NOT USED """
        self.residual_input_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def _init_downsample(self):
        self.down_sample_conv = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x, time_embs):
        """
            Forward function for the DownBlock
        """
        resnet_input = x
        out = self.resnet_conv1(x)
        out = out + self.time_emb_layer1(time_embs)[:, :, None, None]
        out = self.resnet_conv2(out)
        #out = out + self.residual_input_conv(resnet_input)

        batch_size, channels_num , h, w = out.shape
        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms1(input_att)
        input_att = input_att.transpose(1, 2)

        out_att, _ = self.attentions1(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att
        out1 = out 

        out = self.resnet_conv3(out)
        out = out + self.time_emb_layer2(time_embs)[:, :, None, None]
        out = self.resnet_conv4(out)

        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms2(input_att)
        input_att = input_att.transpose(1, 2)
        out_att, _ = self.attentions2(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att
        out2 = out

        out_down = self.down_sample_conv(out)

        return [out1, out2], out_down


class MidBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, att_head_num=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.att_head_num = att_head_num

        self._init_resnet1()
        self._init_time_emb_layer()
        self._init_resnet2()
        self._init_attentions()

    def _init_resnet1(self, group_num=8, dropout_rate=0.):
        self.resnet_conv1 = nn.Sequential(
            nn.GroupNorm(group_num, self.in_channels),
            nn.SiLU(),
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.resnet_conv2 = nn.Sequential(
            nn.GroupNorm(group_num, self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate), # fix: dropout rate is 0.
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )

    def _init_time_emb_layer(self):
        """ Initialize a layer for sinusoidal time embedings """ 
        self.time_emb_layer1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.out_channels)
        )
        self.time_emb_layer2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.out_channels)
        )

    def _init_resnet2(self, group_num=8, dropout_rate=0.): 
        """ Initialize the second resnet, NOTE: find out whether dropout should be used """
        self.resnet_conv3 = nn.Sequential(
            nn.GroupNorm(group_num, self.out_channels),
            nn.SiLU(),
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.resnet_conv4 = nn.Sequential(
            nn.GroupNorm(group_num, self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate), # fix: dropout rate is 0.
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
        
    
    def _init_attentions(self, num_groups=8):
        """ GroupNorm, transpose, and initialize the attention """
        self.attention_norms = nn.GroupNorm(num_groups, self.out_channels)
        self.attentions = nn.MultiheadAttention(self.out_channels, self.att_head_num, batch_first=True)

    def forward(self, x, time_embs):
        # iter 1 
        out = self.resnet_conv1(x)
        out = out + self.time_emb_layer1(time_embs)[:, :, None, None]
        out = self.resnet_conv2(out)

        batch_size, channels_num, h, w = out.shape
        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms(input_att)
        input_att = input_att.transpose(1, 2)

        out_att, _ = self.attentions(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att

        # iter 2
        out = self.resnet_conv3(x)
        out = out + self.time_emb_layer2(time_embs)[:, :, None, None]
        out = self.resnet_conv4(out)

        return out

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim, att_head_num=4):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_emb_dim = time_emb_dim
        self.att_head_num = att_head_num

        self._init_resnet1()
        self._init_time_emb_layer()
        self._init_resnet2()
        self._init_attentions()
        #self._init_residual_input_conv()
        self._init_upsample()


    def _init_resnet1(self, group_num=8, dropout_rate=0.):
        self.resnet_conv1 = nn.Sequential(
            nn.GroupNorm(group_num, self.in_channels*2),
            nn.SiLU(),
            nn.Conv2d(self.in_channels*2, self.in_channels, kernel_size=3, stride=1, padding=1)
        )
        self.resnet_conv2 = nn.Sequential(
            nn.GroupNorm(group_num, self.in_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate), # fix: dropout rate is 0.
            nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1)
        )
    
    def _init_time_emb_layer(self):
        """ Initialize a layer for sinusoidal time embedings """ 
        self.time_emb_layer1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.in_channels)
        )
        self.time_emb_layer2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.out_channels)
        )
    
    def _init_resnet2(self, group_num=8, dropout_rate=0.): 
        """ Initialize the second resnet, NOTE: find out whether dropout should be used """
        self.resnet_conv3 = nn.Sequential(
            nn.GroupNorm(group_num, self.in_channels*2),
            nn.SiLU(),
            nn.Conv2d(self.in_channels*2, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.resnet_conv4 = nn.Sequential(
            nn.GroupNorm(group_num, self.out_channels),
            nn.SiLU(),
            nn.Dropout(dropout_rate), # fix: dropout rate is 0.
            nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=1)
        )
    
    
    def _init_attentions(self, num_groups=8):
        """ GroupNorm, transpose, and initialize the attention """
        self.attention_norms1 = nn.GroupNorm(num_groups, self.in_channels)
        self.attentions1 = nn.MultiheadAttention(self.in_channels, self.att_head_num, batch_first=True)

        self.attention_norms2 = nn.GroupNorm(num_groups, self.out_channels)
        self.attentions2 = nn.MultiheadAttention(self.out_channels, self.att_head_num, batch_first=True)
    
    def _init_residual_input_conv(self):
        """ NOTE: NOT USED """
        self.residual_input_conv = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

    def _init_upsample(self):
        self.up_sample_conv = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x, out_lst, time_embs):
        """
            Forward function for the DownBlock
        """
        resnet_input = x

        #print(f"shape of x before: {x.shape}")
        x = self.up_sample_conv(x)
        #print(f"shape of x: {x.shape}")
        print(x.shape, out_lst[1].shape)
        x1 = torch.cat([x, out_lst[1]], dim=1)
        #print("x1 shape", x1.shape)

        out = self.resnet_conv1(x1)
        out = out + self.time_emb_layer1(time_embs)[:, :, None, None]
        out = self.resnet_conv2(out)
        #out = out + self.residual_input_conv(resnet_input)

        batch_size, channels_num , h, w = out.shape
        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms1(input_att)
        input_att = input_att.transpose(1, 2)

        out_att, _ = self.attentions1(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att

        x2 = torch.cat([out, out_lst[0]], dim=1)
        #print(out.shape, out_lst[0].shape)
        out = self.resnet_conv3(x2)
        out = out + self.time_emb_layer2(time_embs)[:, :, None, None]
        out = self.resnet_conv4(out)

        batch_size, channels_num, h, w = out.shape
        input_att = out.reshape(batch_size, channels_num, h*w)
        input_att = self.attention_norms2(input_att)
        input_att = input_att.transpose(1, 2)
        out_att, _ = self.attentions2(input_att, input_att, input_att)
        out_att = out_att.transpose(1, 2).reshape(batch_size, channels_num, h, w)
        out = out + out_att


        return out
    

class Unet(nn.Module):
    def __init__(self, channels_lst=[32, 64, 128], im_channels=1, time_emb_dim=256):
        super().__init__()

        self.channels_lst = channels_lst
        self.im_channels = im_channels
        self.time_emb_dim = time_emb_dim

        self.down_blocks = []
        for channel in channels_lst:
            self.down_blocks.append(DownBlock(in_channels=channel, out_channels=channel*2, time_emb_dim=256))

        mid_block_channels_num = channels_lst[-1]*2
        self.mid_block = MidBlock(in_channels=mid_block_channels_num, out_channels=mid_block_channels_num, time_emb_dim=256)

        self.up_blocks = []
        for channel in list(reversed(channels_lst)):
            self.up_blocks.append(UpBlock(in_channels=channel * 2 , out_channels=channel, time_emb_dim=256))

        self._init_conv_in()
        self._init_conv_out()
        self._init_time_proj()


    def _init_conv_in(self):
        self.conv_in = nn.Conv2d(self.im_channels, self.channels_lst[0], kernel_size=3, padding=(1, 1))

    def _init_conv_out(self, group_num=8):
        self.norm_out = nn.GroupNorm(group_num, self.channels_lst[0])
        self.conv_out = nn.Conv2d(self.channels_lst[0], self.im_channels, kernel_size=3, padding=(1, 1))
    
    def _init_time_proj(self):
        self.time_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

    def forward(self, x, timesteps):
        out = self.conv_in(x)

        time_embs = get_sinusoidal_embeddings(torch.as_tensor(timesteps).long(), self.time_emb_dim)
        time_embs = self.time_proj(time_embs)

        down_outs = []
        down_out_lst = []
        for down in self.down_blocks:
            down_outs.append(out)
            out_lst, out = down(out, time_embs)
            down_out_lst.append(out_lst)

        out = self.mid_block(out, time_embs) 

        rev_down_outs = list(reversed(down_outs))
        for i, up in enumerate(self.up_blocks):
            out_lst = rev_down_outs[i]
            out = up(out, out_lst, time_embs)
        
        out = self.norm_out(out)
        out = nn.SiLU()(out)
        out = self.conv_out(out)

        return out



        






        

        






