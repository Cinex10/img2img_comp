from datetime import datetime
import os
import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

class SCCLoss(torch.nn.Module):
    def __init__(self):
        super(SCCLoss, self).__init__()

    def forward(self, fakeI, realI, patch_ver=False):
        def batch_ERSMI(I1, I2):
            batch_size = I1.shape[0]
            # channel_size =
            img_size = I1.shape[1] * I1.shape[2] * I1.shape[3]
            # if I2.shape[1] == 1 and I1.shape[1] != 1:
            #     I2 = I2.repeat(1,3, 1, 1)

            def kernel_F(y, mu_list, sigma):
                # tmp_mu = mu_list.view(-1,1).repeat(1, img_size).repeat(,1,1).cuda()  # [81, 784]
                tmp_mu = mu_list.view(-1, 1).repeat(1, img_size).cuda()
                tmp_y = y.view(batch_size,1, -1).repeat(1,81, 1)
                tmp_y = tmp_mu - tmp_y
                mat_L = torch.exp(tmp_y.pow(2) / (2 * sigma ** 2))
                return mat_L

            mu = torch.Tensor([-1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0]).cuda()

            x_mu_list = mu.repeat(9).view(-1, 81)
            y_mu_list = mu.unsqueeze(0).t().repeat(1, 9).view(-1, 81)

            mat_K = kernel_F(I1, x_mu_list, 1)
            mat_L = kernel_F(I2, y_mu_list, 1)

            H1 = ((mat_K.matmul(mat_K.transpose(1,2))).mul(mat_L.matmul(mat_L.transpose(1,2))) / (img_size ** 2)).cuda()
            # h1 = (mat_K.mul(mat_L)).mm(torch.ones(img_size, 1)) / img_size

            H2 = ((mat_K.mul(mat_L)).matmul((mat_K.mul(mat_L)).transpose(1,2)) / img_size).cuda()
            h2 = ((mat_K.sum(2).view(batch_size,-1, 1)).mul(mat_L.sum(2).view(batch_size,-1, 1)) / (img_size ** 2)).cuda()
            # h2 = (((mat_K.sum(1).view(-1,1)).mul(mat_L.sum(1).view(-1,1)) / (img_size ** 2)).double()).cuda()

            H2 = 0.5 * H1 + 0.5 * H2
            tmp = H2 + 0.05 * torch.eye(H2.shape[1]).cuda()

            alpha = (tmp.inverse())

            alpha = alpha.matmul(h2)
            # end = time.clock()
            # print('2:', end - start)
            ersmi = (2 * (alpha.transpose(1,2)).matmul(h2) - ((alpha.transpose(1,2)).matmul(H2)).matmul(alpha) - 1).squeeze()
            ersmi = -ersmi.mean()
            return ersmi


        batch_loss = batch_ERSMI(fakeI, realI)

        return batch_loss

class Gen(torch.nn.Module):
    def __init__(self,netG_A,netG_B):
        super().__init__()
        self.netG_A = netG_A
        self.netG_B = netG_B

class Desc(torch.nn.Module):
    def __init__(self,netD_A,netD_B):
        super().__init__()
        self.netD_A = netD_A
        self.netD_B = netD_B

class DescPixel(torch.nn.Module):
    def __init__(self,netD_A_pixel,netD_B_pixel):
        super().__init__()
        self.netD_A_pixel = netD_A_pixel
        self.netD_B_pixel = netD_B_pixel

class SCCCycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' discriminator (PatchGAN introduced by pix2pix),
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_SCC', type=float, default=0.9, help='lambda SCC')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_GAN_A', type=float, default=1.0, help='weight for GAN loss (A -> B )')
            parser.add_argument('--lambda_GAN_B', type=float, default=1.0, help='weight for GAN loss (B -> A)')
        return parser

    def __init__(self, opt,fabric):
        """Initialize the SCCCycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        self.fabric = fabric
        self.a_b_infrence_times = []
        self.b_a_infrence_times = []
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'SCC' ,'idt_B']
        self.loss_names.append('D_A_pixel')
        self.loss_names.append('D_B_pixel')
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)
        netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.gen = Gen(netG_A=netG_A,netG_B=netG_B)
        
        if self.isTrain:  # define discriminators
            netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.desc = Desc(netD_A=netD_A,netD_B=netD_B)
            
            netD_A_pixel = networks.define_D(opt.output_nc, opt.ndf, 'pixel',
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            netD_B_pixel = networks.define_D(opt.input_nc, opt.ndf, 'pixel',
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
            
            self.desc_pixel = DescPixel(netD_A_pixel=netD_A_pixel,netD_B_pixel=netD_B_pixel)
            
        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionSCC = SCCLoss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.gen.netG_A.parameters(), self.gen.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.desc.netD_A.parameters(), self.desc.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_pixel = torch.optim.Adam(itertools.chain(self.desc_pixel.netD_A_pixel.parameters(), self.desc_pixel.netD_B_pixel.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            
            if self.opt.bf16:
                self.gen,self.optimizer_G = self.fabric.setup(self.gen,self.optimizer_G) 
                self.desc,self.optimizer_D = self.fabric.setup(self.desc,self.optimizer_D) 
                self.desc_pixel,self.optimizer_D_pixel = self.fabric.setup(self.desc_pixel,self.optimizer_D_pixel) 
            
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def print_networks(self, verbose):
            """Print the total number of parameters in the network and (if verbose) network architecture

            Parameters:
                verbose (bool) -- if verbose: print the network architecture
            """
            print('---------- Networks initialized -------------')
            for name in self.model_names:
                if isinstance(name, str):
                    #net = getattr(self, 'net' + name)
                    net = self.get_correct_attr(name)
                    num_params = 0
                    for param in net.parameters():
                        num_params += param.numel()
                    if verbose:
                        print(net)
                    print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
            print('-----------------------------------------------')

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = self.get_correct_attr(name)
                net.eval()

    def save_networks(self, epoch):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                net = self.get_correct_attr(name)

                if len(self.gpu_ids) > 0 and torch.cuda.is_available():
                    torch.save(net.module.cpu().state_dict(), save_path)
                    net.cuda(self.gpu_ids[0])
                else:
                    torch.save(net.cpu().state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.save_dir, load_filename)
                net = self.get_correct_attr(name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=str(self.device))
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))
                net.load_state_dict(state_dict)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        start = datetime.now()
        self.fake_B = self.gen.netG_A(self.real_A)  # G_A(A)
        end = datetime.now()
        self.a_b_infrence_times.append((end - start).total_seconds())
        self.rec_A = self.gen.netG_B(self.fake_B)   # G_B(G_A(A))
        start = datetime.now()
        self.fake_A = self.gen.netG_B(self.real_B)  # G_B(B)
        end = datetime.now()
        self.b_a_infrence_times.append((end - start).total_seconds())
        self.rec_B = self.gen.netG_A(self.fake_A)   # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        if self.opt.bf16:
            self.fabric.backward(loss_D)
        else:
            loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.desc.netD_A, self.real_B, fake_B)
        self.loss_D_A_pixel = self.backward_D_basic(self.desc_pixel.netD_A_pixel, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.desc.netD_B, self.real_A, fake_A)
        self.loss_D_B_pixel = self.backward_D_basic(self.desc_pixel.netD_B_pixel,  self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        lambda_SCC = self.opt.lambda_SCC
        lambda_GAN_A = self.opt.lambda_GAN_A
        lambda_GAN_B = self.opt.lambda_GAN_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            self.idt_A = self.gen.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            self.idt_B = self.gen.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.desc.netD_A(self.fake_B), True) * lambda_GAN_A
        self.loss_G_A += self.criterionGAN(self.desc_pixel.netD_A_pixel(self.fake_B), True) * lambda_GAN_A
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.desc.netD_B(self.fake_A), True) * lambda_GAN_B
        self.loss_G_B += self.criterionGAN(self.desc_pixel.netD_B_pixel(self.fake_A), True) * lambda_GAN_B
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        #MI loss
        if lambda_SCC == 0 :
            self.loss_SCC = 0
        else: 
            self.loss_SCC_A = self.criterionSCC(self.real_A, self.fake_B) * lambda_SCC
            self.loss_SCC_B = self.criterionSCC(self.real_B, self.fake_A) * lambda_SCC
            self.loss_SCC = self.loss_SCC_A + self.loss_SCC_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_SCC
        if self.opt.bf16:
            self.fabric.backward(self.loss_G)
        else:
            self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.desc.netD_A, self.desc.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.set_requires_grad([self.desc_pixel.netD_A_pixel, self.desc_pixel.netD_B_pixel], False)
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.desc.netD_A, self.desc.netD_B], True)
        self.set_requires_grad([self.desc_pixel.netD_A_pixel, self.desc_pixel.netD_B_pixel], True)
        self.optimizer_D_pixel.zero_grad()
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
        self.optimizer_D_pixel.step()
    
    def get_correct_attr(self,name):
        match name[0]:
            case 'G':
                return getattr(self.gen, 'net' + name)            
            case 'D':
                return getattr(self.desc, 'net' + name)
            #case 'F':
            #    return getattr(self.f, 'net' + name)          