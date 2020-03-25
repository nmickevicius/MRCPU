import numpy as np
import matplotlib.pyplot as plt
import sigpy as sp
import pywt

class Aclass(object):
    def __init__(self, F, phi, smaps):
        self.F = F
        self.phi = phi
        smaps = np.reshape(np.copy(smaps),(smaps.shape[0],smaps.shape[1],smaps.shape[2],1))
        smaps = np.transpose(smaps,(0,3,1,2))
        self.smaps = smaps
    def A(self,img):
        ''' Forward operator '''
        # img - [npathways,nsegments,nreadout]
        # ks - [coils,npathways,segments,nreadout]
        
        # multiply by coil sensitivity maps
        img = np.reshape(np.copy(img),(img.shape[0],img.shape[1],img.shape[2],1))
        img = np.transpose(img,(3,0,1,2))
        img = img * self.smaps
        
        # bring to k-space
        ktmp = self.F * img
        
        # modulate by pathway phase
        ks = np.zeros(ktmp.shape,dtype=ktmp.dtype)
        for pw in range(self.phi.shape[0]):
            ks[::,pw,::,::] = ktmp[::,pw,::,::]
            for opw in range(self.phi.shape[0]):
                if opw != pw:
                    tmp = ktmp[::,opw,::,::]
                    phitmp = np.copy(self.phi[opw,::,::])
                    phitmp = np.transpose(np.reshape(phitmp,(phitmp.shape[0],phitmp.shape[1],1)),(2,0,1))
                    tmp = tmp * phitmp # add phase applied to current pathway
                    phitmp = np.copy(self.phi[pw,::,::])
                    phitmp = np.transpose(np.reshape(phitmp,(phitmp.shape[0],phitmp.shape[1],1)),(2,0,1))
                    tmp = tmp * np.conj(phitmp) # add the conjugate of the phase
                    ks[::,pw,::,::] += tmp
    
        return ks
    def At(self,ks):
        ''' Transpose operator '''
        # ks - [coils,npathways,nsegments,nreadout]
        # img - [npathways,nsegments,nreadout]
        img = self.F.H * ks
        img = np.sum(img * np.conj(self.smaps),axis=0)
        return img
    def AtA(self,inp):
        return self.At(self.A(inp))
    def AtA_admm_cg(self,inp,rho):
        return rho*inp + self.At(self.A(inp))
    def __call__(self,inp):
        return self.AtA(inp)

def prepPhi(data,phi):
    # data - [coils,segments,readout]
    # out - [coils,pathways,segments,readout]
    out = np.zeros((data.shape[0],phi.shape[0],data.shape[1],data.shape[2]),dtype=data.dtype)
    for pw in range(phi.shape[0]):
        phitmp = np.copy(phi[pw,::,::])
        phitmp = np.transpose(np.reshape(phitmp,(phitmp.shape[0],phitmp.shape[1],1)),(2,0,1))
        out[::,pw,::,::] = np.copy(data) * np.conj(phitmp)
    return out

def admm_cg_subproblem(A,rhs,cgiter,rho,x0):

    # x = np.zeros(rhs.shape,rhs.dtype)
    x = np.copy(x0)
    d = rhs - A.AtA_admm_cg(x,rho)
    r = np.copy(d)
    normrr0 = np.sum(np.conj(rhs)*rhs, axis=None)
    normrr = np.sum(np.conj(r)*r,axis=None)
    for n in range(cgiter):
        AtAd = A.AtA_admm_cg(d,rho)
        tmp = np.conj(d) * AtAd
        alpha = normrr / np.sum(tmp,axis=None)
        x = x + alpha*d
        r = r - alpha*AtAd
        normrr2 = np.sum(np.conj(r)*r,axis=None)
        beta = normrr2/normrr
        normrr = normrr2
        d = r + beta*d
        print('\tCG iteration #%i: r/r0 = %f'%(n,np.sqrt(np.abs(normrr)/np.abs(normrr0))))
    return x

def im2col(im, W):
    rows,cols = im.shape
    npr = int(rows/W)
    npc = int(cols/W)
    out = np.zeros((int(W*W),int(npr*npc)),dtype=im.dtype)
    idx = 0
    for ipr in range(npr):
        ir1 = int(ipr*W)
        ir2 = int(ir1+W)
        for ipc in range(npc):
            ic1 = int(ipc*W)
            ic2 = int(ic1+W)
            patch = im[ir1:ir2,ic1:ic2]
            out[::,idx] = np.reshape(patch,(-1))
            idx += 1
    return out

def col2im(v, W, rows, cols):
    npr = int(rows/W)
    npc = int(cols/W)
    out = np.zeros((rows,cols),dtype=v.dtype)
    idx = 0
    for ipr in range(npr):
        ir1 = int(ipr*W)
        ir2 = int(ir1+W)
        for ipc in range(npc):
            ic1 = int(ipc*W)
            ic2 = int(ic1+W)
            out[ir1:ir2,ic1:ic2] = np.reshape(v[::,idx],(W,W))
            idx += 1
    return out

def SoftThresh(y,lam):
    xhat = y * (np.abs(y) - lam)/np.abs(y)
    xhat[np.abs(y)<=lam] = 0.0
    return xhat

def wavMask(dims, scale):
    sx, sy = dims
    res = np.ones(dims)
    NM = np.round(np.log2(dims))
    for n in range(int(np.min(NM)-scale+2)//2):
        res[:int(np.round(2**(NM[0]-n))), :int(np.round(2**(NM[1]-n)))] = \
            res[:int(np.round(2**(NM[0]-n))), :int(np.round(2**(NM[1]-n)))]/2
    return res

def coeffs2img(LL, coeffs):
    LH, HL, HH = coeffs
    return np.vstack((np.hstack((LL, LH)), np.hstack((HL, HH))))

def unstack_coeffs(Wim):
        L1, L2  = np.hsplit(Wim, 2)
        LL, HL = np.vsplit(L1, 2)
        LH, HH = np.vsplit(L2, 2)
        return LL, [LH, HL, HH]


def img2coeffs(Wim, levels=4):
    LL, c = unstack_coeffs(Wim)
    coeffs = [c]
    for i in range(levels-1):
        LL, c = unstack_coeffs(LL)
        coeffs.insert(0,c)
    coeffs.insert(0, LL)
    return coeffs


def dwt2(im):
    coeffs = pywt.wavedec2(im, wavelet='db4', mode='per', level=4)
    Wim, rest = coeffs[0], coeffs[1:]
    for levels in rest:
        Wim = coeffs2img(Wim, levels)
    return Wim


def idwt2(Wim):
    coeffs = img2coeffs(Wim, levels=4)
    return pywt.waverec2(coeffs, wavelet='db4', mode='per')


def admm(data, F, phi, smaps, admmiter, cgiter, rho, lam):

    A = Aclass(F, phi, smaps)
    y = prepPhi(data,phi)
    b = A.At(y)
    
    lam = lam * np.max(np.abs(b))

    x = np.copy(b)
    z = np.zeros(x.shape,dtype=x.dtype)
    u = np.zeros(x.shape,dtype=x.dtype)

    for ii in range(admmiter):

        print('ADMM Iteration #%i:'%(ii))

        # conjugate gradient subproblem
        x = admm_cg_subproblem(A, b+rho*(z-u), cgiter, rho, x)

        # update z using soft thresholding in wavelet domain
        xpu = x + u
        z = np.zeros(xpu.shape,xpu.dtype)
        for pw in range(xpu.shape[0]):
            tmp = dwt2(xpu[pw,::,::])
            tmp = SoftThresh(tmp, lam/rho)
            z[pw,::,::] = idwt2(tmp)


        # update u
        u = xpu - z

    return x
