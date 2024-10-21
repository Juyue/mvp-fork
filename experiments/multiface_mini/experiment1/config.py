# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import data.multiviewvideo as datamodel

def get_dataset(camerafilter=lambda x: True, maxframes=-1, subsampletype=None):
    root_dir = "/root/datasets/multiface_mini/m--20180227--0000--6795937--GHS"
    return datamodel.Dataset(
        krtpath=os.path.join(root_dir, "KRT"),
        geomdir=root_dir,
        imagepath=os.path.join(root_dir, "images/{seg}/{cam}/{frame:06d}.png"),
        keyfilter=["image", "pixelcoords", "verts", "tex", "avgtex", "camera"],
        camerafilter=camerafilter,
        segmentfilter=lambda x: x.startswith("E057"),
        maxframes=maxframes,
        subsampletype=subsampletype,
    )

def get_autoencoder(dataset):
    # import models.colorcals.colorcal1 as colorcalib
    import models.volumetric as aemodel
    import models.encoders.geotex as encoderlib
    import models.decoders.mvp as decoderlib
    import models.raymarchers.mvpraymarcher as raymarcherlib
    import models.colorcals as colorcalib
    import utils.utils as utils
    
    obj_data_path = os.path.join(dataset.geomdir, "tracked_mesh", dataset.seg, f"{dataset.frame:06d}.obj")
    v, vt, vindices, vtindices = utils.load_obj(obj_data_path)
    idxim, tidxim, barim = utils.gentritex(v, vt, vindices, vtindices, 1024)
    vertmean, vertstd = dataset.vertmean, dataset.vertstd

    volradius = 1.0

    return aemodel.Autoencoder(
        dataset,
        encoder=encoderlib.Encoder(),
        decoder=decoderlib.Decoder(
            vt=vt,
            vertmean=vertmean,
            vertstd=vertstd,
            idxim=idxim,
            tidxim=tidxim,
            barim=barim,
            volradius=volradius,
            nprims=4096,
            primsize=(32, 32, 32),
        ),
        raymarcher=raymarcherlib.Raymarcher(volradius=volradius),
        colorcal=colorcalib.Colorcal(dataset.allcameras),
        volradius=volradius
        )

### profiles
# A profile is instantiated by the training or evaluation scripts
# and controls how the dataset and autoencoder is created
class Train():
    batchsize=4
    maxiter=500000
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_dataset(self): return get_dataset()
    def get_optimizer(self, ae):
        import itertools
        import torch.optim
        lr = 0.0001
        aeparams = itertools.chain(
            [{"params": x} for x in ae.encoder.parameters()],
            [{"params": x} for x in ae.decoder.parameters()],
            [{"params": x} for x in ae.colorcal.parameters()])
        return torch.optim.Adam(aeparams, lr=lr, betas=(0.9, 0.999))
    def get_loss_weights(self):
        return {"kldiv": 0.01,
                "primvolsum": 0.01,
                # "vertmse": 0.01,
                "irgbmse": 1.0,
                }
    def get_outputlist(self):
        return ["irgbrec"]
    
    def get_ae_args(self):
        return {"renderoptions": {"dt": 0.01}}

class ProgressWriter():
    def batch(self, iternum, itemnum, **kwargs):
        import numpy as np
        from PIL import Image
        rows = []
        row = []
        for i in range(kwargs["image"].size(0)):
            row.append(
                np.concatenate((
                    kwargs["irgbrec"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2],
                    kwargs["image"][i].data.to("cpu").numpy().transpose((1, 2, 0))[::2, ::2]), axis=1))
            if len(row) == 2:
                rows.append(np.concatenate(row, axis=1))
                row = []
        imgout = np.concatenate(rows, axis=0)
        outpath = os.path.dirname(__file__)
        Image.fromarray(np.clip(imgout, 0, 255).astype(np.uint8)).save(os.path.join(outpath, "prog_{:06}.jpg".format(iternum)))

class Progress():
    """Write out diagnostic images during training."""
    batchsize=2

    def get_ae_args(self): return dict(outputlist=["irgbrec"], renderoptions={"dt": 0.01})
    def get_dataset(self): return get_dataset(maxframes=1)
    def get_writer(self): return ProgressWriter()
    def get_outputlist(self):
        return ["irgbrec"]

class Render():
    """Render model with training camera or from novel viewpoints.
    
    e.g., python render.py {configpath} Render --maxframes 128"""
    def __init__(self, cam=None, maxframes=-1, showtarget=False, viewtemplate=False):
        self.cam = cam
        self.maxframes = maxframes
        self.showtarget = showtarget
        self.viewtemplate = viewtemplate
    def get_autoencoder(self, dataset): return get_autoencoder(dataset)
    def get_ae_args(self): return dict(outputlist=["irgbrec"], viewtemplate=self.viewtemplate)
    def get_dataset(self):
        import data.utils
        import eval.cameras.rotate as cameralib
        dataset = get_dataset(camerafilter=lambda x: x == self.cam, maxframes=self.maxframes)
        if self.cam is None:
            camdataset = cameralib.Dataset(len(dataset))
            return data.utils.JoinDataset(camdataset, dataset)
        else:
            return dataset
    def get_writer(self):
        import eval.writers.videowriter as writerlib
        return writerlib.Writer(
            os.path.join(os.path.dirname(__file__),
                "render_{}{}.mp4".format(
                    "rotate" if self.cam is None else self.cam,
                    "_template" if self.viewtemplate else "")),
            showtarget=self.showtarget)