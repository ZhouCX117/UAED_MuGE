import os
from shutil import copy
src_dir="/data/zhoucaixia/workspace/UD_Edge/tmp/trainval_sigma_logit_unetpp_alpha_ffthalf_feat_testalpha_clipsum/alpha_style_all_epoch19//"
dst_dir=os.path.join(src_dir+'best_ois_0.1/png')
os.makedirs(dst_dir,exist_ok=True)
record=open("/data/zhoucaixia/workspace/UD_Edge/tmp/trainval_sigma_logit_unetpp_alpha_ffthalf_feat_testalpha_clipsum/alpha_style_all_epoch19//record.txt","r").readlines()

model_name=['0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1']
model_name_dict={}
for i in range(len(model_name)):
    model_name_dict[i]=model_name[i]
print(model_name_dict)
for index in range(len(record)):
    line=record[index].strip("\n").split("\t")
    name=line[0]
    # kind=int(line[1])/(len(model_name)-1)
    # if kind==0 or kind==1:
    #     kind=int(kind)
    src_file_name=os.path.join(src_dir+model_name_dict[int(line[1])],name+".png")
    dsc_file_name=os.path.join(dst_dir,name+".png")
    copy(src_file_name,dsc_file_name)
    print(src_file_name)
