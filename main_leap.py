from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader_leap_add_label import *
from main_network_temporal_add_label import AVVPNet
from utils.eval_metrics import segment_level, event_level
import pandas as pd


def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)



def temporal_visual_loss(temporal_pred_labels, temporal_pseudo_visual_labels):
    '''
    segment-level loss, directly compute the binary cross entropy loss of the prediction and the pseudo labels.
    '''
    # inputs: [bs, 10, 25], after sigmoid
    loss = nn.BCELoss()(temporal_pred_labels, temporal_pseudo_visual_labels.float())
    return loss



def avss_loss(output, a_prob, v_prob, a_fea, v_fea, a_frame_prob, v_frame_prob, audio_pseudo_labels, visual_pseudo_labels, eps=0.01, use_pseudo_label=True):
    if not use_pseudo_label:
        o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)
        oa = (a_prob.cpu().detach().numpy() >= 0.5).astype(np.int_) # (B, 25)
        ov = (v_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
        
        oa = np.expand_dims(oa, axis=1)
        ov = np.expand_dims(ov, axis=1) # (B, 1, 25)

        Pa = a_frame_prob.cpu().detach().numpy()  # (B, 10, 25)
        Pv = v_frame_prob.cpu().detach().numpy()  # (B, 10, 25)

        Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(oa, repeats=10, axis=1)
        Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(ov, repeats=10, axis=1)
        Pa = torch.from_numpy(Pa).to(a_fea.device)
        Pv = torch.from_numpy(Pv).to(v_fea.device)
    else: #! default
        Pa = audio_pseudo_labels.float()  # (B, 10, 25)
        Pv = visual_pseudo_labels # (B, 10, 25)
    P_av = Pa * Pv # (B, 10, 25)
    a_event_num = torch.max(torch.sum(Pa, dim=-1), dim=-1)[0] # (B,)
    v_event_num = torch.max(torch.sum(Pv, dim=-1), dim=-1)[0]
    max_event_num = torch.zeros_like(a_event_num).float().to('cuda')
    for i in range(a_event_num.shape[0]):
        if a_event_num[i] >= v_event_num[i]:
            max_event_num[i] = a_event_num[i]
        else:
            max_event_num[i] = v_event_num[i]
    for i in range(a_event_num.shape[0]):
        if max_event_num[i] == 0:
            max_event_num[i] += 1e-8
    avc_label = torch.bmm(Pa, Pv.permute(0, 2, 1)) / max_event_num.unsqueeze(-1).unsqueeze(-1)

    a_fea = F.normalize(a_fea, dim=-1)
    v_fea = F.normalize(v_fea, dim=-1) # (B, T, C)

    avc_pred = torch.bmm(a_fea, v_fea.permute(0, 2, 1)) # (B, 10, 10)
    avc_label = torch.sigmoid(torch.log(avc_label)) + eps
    loss = nn.MSELoss()(F.relu(avc_pred), avc_label)
    return loss




def train(args, model, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample[
            'video_st'].to('cuda'), sample['label'].type(torch.FloatTensor).to('cuda')
        temporal_pseudo_visual_labels = sample['temporal_pv'].float().to('cuda') # [bs, 10, 25]
        temporal_pseudo_audio_labels = sample['temporal_pa'].float().to('cuda') # [bs, 10, 25]

        label_embed = sample['label_text_embed'].to('cuda') # [bs, 25, 300/768/clip]

        optimizer.zero_grad()
        output, a_prob, v_prob, a_frame_prob, v_frame_prob, a_fea, v_fea, la_f, lv_f = model(audio, video, video_st, label_embed)
        output.clamp_(min=1e-7, max=1 - 1e-7)
        a_prob = a_prob.clamp(min=1e-7, max=1 - 1e-7)
        v_prob = v_prob.clamp(min=1e-7, max=1 - 1e-7)
        a_frame_prob = a_frame_prob.clamp(min=1e-7, max=1 - 1e-7)
        v_frame_prob = v_frame_prob.clamp(min=1e-7, max=1 - 1e-7)

        Pa = sample['pa'].type(torch.FloatTensor).to('cuda') # 
        Pv = sample['pv'].type(torch.FloatTensor).to('cuda')
        
        # individual guided learning
        loss1 = torch.Tensor([0])
        loss2 = torch.Tensor([0])
        loss3 = torch.Tensor([0])
        loss1 =  criterion(a_prob, Pa) 
        loss2 =  criterion(v_prob, Pv) 
        loss3 =  criterion(output, target) 
        loss = loss1 + loss2 + loss3

        loss4 = torch.Tensor([0])
        if args.temporal_v_loss_flag:
            loss4 = temporal_visual_loss(v_frame_prob, temporal_pseudo_visual_labels)
            loss += args.loss_temporal_wei * loss4
        
        loss5 = torch.Tensor([0])
        if args.temporal_a_loss_flag:
            loss5 = temporal_visual_loss(a_frame_prob, temporal_pseudo_audio_labels)
            loss += args.loss_temporal_wei * loss5

        loss6 = torch.Tensor([0])
        if args.avss_loss_flag:
             #! the proposed audio-visual semantic similarity loss
            loss6 = avss_loss(output, a_prob, v_prob, a_fea, v_fea, a_frame_prob, v_frame_prob, temporal_pseudo_audio_labels, temporal_pseudo_visual_labels, use_pseudo_label=args.use_pseudo_label, eps=args.eps)
            loss +=  args.loss_avss_wei * loss6


        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss1: {:.3f} Loss2: {:.3f} Loss3: {:.3f} Loss4: {:.3f} Loss5: {:.3f} Loss6: {:.4f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss1.item(),  loss2.item(), loss3.item(), loss4.item(), loss5.item(), loss6.item()))


def eval(model, val_loader, set):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    model.eval()

    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    data_dir = '../data/'
    df_a = pd.read_csv(data_dir + "AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv(data_dir + "AVVP_eval_visual.csv", header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample[
                'video_st'].to('cuda'), sample['label'].to('cuda')
            label_embed = sample['label_text_embed'].to('cuda') # [bs, 25, 300/768/clip]
            video_name = sample['video_name']
            temporal_pseudo_visual_labels = sample['temporal_pv'].to('cuda') # [bs, 10, 25]
            temporal_pseudo_audio_labels = sample['temporal_pa'].to('cuda') # [bs, 10, 25]

            output, a_prob, v_prob, a_frame_prob, v_frame_prob, a_fea, v_fea, la_f, lv_f  = model(audio, video, video_st, label_embed)
            o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)
            oa = (a_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            ov = (v_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)

            Pa = a_frame_prob[0].cpu().detach().numpy()
            Pv = v_frame_prob[0].cpu().detach().numpy()
            # filter out false positive events with predicted weak labels
            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(o, repeats=10, axis=0)

            # extract audio GT labels
            GT_a = np.zeros((25, 10))
            GT_v = np.zeros((25, 10))

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1

            # extract visual GT labels
            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1

            GT_av = GT_a * GT_v

            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    print('Audio Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_a))))
    print('Visual Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_v))))
    print('Audio-Visual Event Detection Segment-level F1: {:.1f}'.format(100 * np.mean(np.array(F_seg_av))))

    avg_type = (100 * np.mean(np.array(F_seg_av)) + 100 * np.mean(np.array(F_seg_a)) + 100 * np.mean(
        np.array(F_seg_v))) / 3.
    avg_event = 100 * np.mean(np.array(F_seg))
    print('Segment-levelType@Avg. F1: {:.1f}'.format(avg_type))
    print('Segment-level Event@Avg. F1: {:.1f}'.format(avg_event))

    print('Audio Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_a))))
    print('Visual Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_v))))
    print('Audio-Visual Event Detection Event-level F1: {:.1f}'.format(100 * np.mean(np.array(F_event_av))))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(
        np.array(F_event_v))) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))
    print('Event-level Type@Avg. F1: {:.1f}'.format(avg_type_event))
    print('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))
    return avg_type



def main():
    # Training settings
    data_dir = '../data/'
    parser = argparse.ArgumentParser(description='PyTorch Implementation of MM_Pyramid')
    parser.add_argument(
        "--audio_dir", type=str, default='../data/feats/vggish/', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='../data/feats/res152/',
        help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='../data/feats/r2plus1d_18/',
        help="video dir")
    parser.add_argument(
        "--label_train", type=str, default=data_dir + "AVVP_train.csv", help="weak train csv file")
    parser.add_argument(
        "--label_val", type=str, default=data_dir + "AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument(
        "--label_test", type=str, default=data_dir + "AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch_size', type=int, default=16, metavar='N',
                        help='input batch size for training (default: 16)')
    
    parser.add_argument('--mmp_head', type=int, default=8,
                        help='head number of multi-head attention in MM-Pry Encoder')
    parser.add_argument('--avl_head', type=int, default=8,
                        help='head number of multi-head attention in LEAP Decoder')

    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--stepsize', type=int, default=10,
                        help='step size of learning scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='gamma of learning scheduler')
    parser.add_argument(
        "--model", type=str, default='MM_Pyramid', help="with backbone to use")
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')

    parser.add_argument('--levels', type=int, default=4, help='levels of TCN in MM-Pyr')
    parser.add_argument('--hid_dim', type=int, default=512, help='number of hidden units per layer')
    parser.add_argument('--ffn_dim', type=int, default=512, help='dimension of feed forward layer')

    parser.add_argument("--v_pseudo_flag", action='store_true', default=False, help="use visual pseudo label generated by VALOR")
    parser.add_argument("--a_pseudo_flag", action='store_true', default=False, help="use audio pseudo label generated by VALOR")

    parser.add_argument('--temporal_v_loss_flag', action="store_true", default=False, help="segment-level visual loss")
    parser.add_argument('--temporal_a_loss_flag', action="store_true", default=False, help="segment-level audio loss")
    parser.add_argument('--loss_temporal_wei', type=float, default=0.5, help='weight for segment-level loss')

    parser.add_argument('--avss_loss_flag', action="store_true", default=False, help="the proposed avss loss")
    parser.add_argument('--loss_avss_wei', type=float, default=0.5, help='weight for avss loss')
    parser.add_argument('--eps', type=float, default=0.01, help='weight for avss loss')
    parser.add_argument("--use_pseudo_label", action="store_true", help="whether use pseudo label to index features when computing avss loss")

    parser.add_argument("--dataset_label_embedding_path", type=str, default='./glove/6B/300d/llp_label_glove_embeddings.pt', help="label embedding path")
    parser.add_argument("--dataset_word_embed_dim", type=int, default=300, help="[300 for glove] [768 for bert]")
    # parser.add_argument("--dataset_label_embedding_path", type=str, default='./bert/wo_preprompt/llp_label_bert_embeddings.pt', help="label embedding path")
    # parser.add_argument("--dataset_word_embed_dim", type=int, default=768, help="[300 for glove] [768 for bert]")

    parser.add_argument('--lv_layer_num', type=int, default=2, help='number of hidden units per layer')
    parser.add_argument('--la_layer_num', type=int, default=2, help='number of hidden units per layer')

    args = parser.parse_args()
    print('-' * 30)
    for k in list(vars(args).keys()):
        print('%s: %s' % (k, vars(args)[k]))
    print('-' * 30)


    if args.v_pseudo_flag:
        print("[SETTING] v_pseudo_flag = True")
    else:
        print("[SETTING] v_pseudo_flag = False")
    if args.a_pseudo_flag:
        print("[SETTING] a_pseudo_flag = True")
    else:
        print("[SETTING] a_pseudo_flag = False")


    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    seed_everything(args.seed)
    n_channels = [args.hid_dim] * args.levels
    model = AVVPNet(args.mmp_head, args.avl_head, args.hid_dim, args.ffn_dim, n_channels, args.dataset_word_embed_dim, args.lv_layer_num, args.la_layer_num).to('cuda')


    if args.mode == 'train':
        train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                    st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]),
                                    dataset_label_embedding_dir=args.dataset_label_embedding_path, word_embedding_dim=args.dataset_word_embed_dim,
                                    v_pseudo_flag=args.v_pseudo_flag, a_pseudo_flag=args.a_pseudo_flag)
        val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                  st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]), 
                                   dataset_label_embedding_dir=args.dataset_label_embedding_path, word_embedding_dim=args.dataset_word_embed_dim,
                                    v_pseudo_flag=args.v_pseudo_flag, a_pseudo_flag=args.a_pseudo_flag)
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, transform = transforms.Compose([ToTensor()]),
                                   dataset_label_embedding_dir=args.dataset_label_embedding_path, word_embedding_dim=args.dataset_word_embed_dim,
                                    v_pseudo_flag=args.v_pseudo_flag, a_pseudo_flag=args.a_pseudo_flag)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=args.gamma)
        criterion = nn.BCELoss()
        best_F = 0
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            scheduler.step(epoch)
            F = eval(model, val_loader, args.label_val)
            if F >= best_F:
                best_F = F
                torch.save(model.state_dict(), args.model_save_dir + '/' + args.model + ".pt")
                print(">>> model saved at ", args.model_save_dir + '/'+ args.model + ".pt")
                print("\n\n")
    elif args.mode == 'val':
        test_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                   st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]), 
                                   dataset_label_embedding_dir=args.dataset_label_embedding_path, word_embedding_dim=args.dataset_word_embed_dim,
                                    v_pseudo_flag=args.v_pseudo_flag, a_pseudo_flag=args.a_pseudo_flag)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + args.model + ".pt"))
        eval(model, test_loader, args.label_val)
    else:
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                   st_dir=args.st_dir, transform=transforms.Compose([ToTensor()]),
                                   dataset_label_embedding_dir=args.dataset_label_embedding_path, word_embedding_dim=args.dataset_word_embed_dim,
                                    v_pseudo_flag=args.v_pseudo_flag, a_pseudo_flag=args.a_pseudo_flag)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        model.load_state_dict(torch.load(args.model_save_dir + '/' + args.model + ".pt"))
        eval(model, test_loader, args.label_test)


if __name__ == '__main__':
    main()
