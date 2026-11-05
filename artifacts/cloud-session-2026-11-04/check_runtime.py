"""Cloud library and CUDA operator check. No model construction, weights, or training."""
import importlib.metadata as metadata
import json, pathlib, sys, socket
import torch, torchvision, ultralytics, cv2, PIL, matplotlib, polars
root=pathlib.Path('/root/rivermind-data/FIRE_R2')
assert sys.prefix==str(root/'.envs/sci-env')
assert socket.gethostname()=='jupyter-7lrm3ongvydsr1db'
packages={d.metadata['Name']:d.version for d in metadata.distributions()}
x=torch.tensor([1.,2.,3.],device='cuda')
assert (x*2+1).tolist()==[3.,5.,7.]
boxes=torch.tensor([[0.,0.,4.,4.],[0.,0.,3.,3.]],device='cuda')
keep=torchvision.ops.nms(boxes,torch.tensor([0.9,0.8],device='cuda'),0.4).cpu().tolist()
assert keep==[0]
provenance=json.loads(metadata.distribution('ultralytics').read_text('direct_url.json'))
report={'hostname':socket.gethostname(),'prefix':sys.prefix,'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,'opencv':cv2.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'cuda_arithmetic_passed':True,'cuda_nms_passed':True,'packages':packages,'ultralytics_provenance':provenance,'model_constructed':False,'weights_downloaded_or_loaded':False,'training_started':False}
(root/'artifacts/cloud-runtime-check.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='packages'},indent=2))
