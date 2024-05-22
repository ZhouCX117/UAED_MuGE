import os
import numpy as np
import sys
from scipy.interpolate import interp1d
from utils_ois import computeRPF,computeRPF_numpy,findBestRPF
record_txt=open("/data/zhoucaixia/workspace/UD_Edge/tmp/trainval_sigma_logit_unetpp_alpha_ffthalf_feat_testalpha_clipsum/alpha_style_all_epoch19/record.txt",'w')

eps=sys.float_info.epsilon
np.seterr(divide='ignore',invalid='ignore')
Z=np.zeros((100,1))
from glob import glob
cntR_all,sumR_all,cntP_all,sumP_all=Z,Z,Z,Z
T=np.array([i/100 for i in range(1,100)])
oisCntR,oisSumR,oisCntP,oisSumP=0,0,0,0
def index_select(arr1,index_arr):
    return_arr=np.zeros((100,1))
    for i in range(len(index_arr)):
        return_arr[i][0]=arr1[index_arr[i]][i]
    
    return return_arr
exp_dir='/data/zhoucaixia/workspace/UD_Edge/tmp/trainval_sigma_logit_unetpp_alpha_ffthalf_feat_testalpha_clipsum/alpha_style_all_epoch19/'
model_name=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

eval_list=glob(os.path.join(exp_dir+"0","nms-eval/*_ev1.txt"))

best_ods_dir=os.path.join(exp_dir+"best_ods_0.1/nms-eval")
os.makedirs(best_ods_dir,exist_ok=True)

for eval_name in eval_list:
   
    eval=eval_name.split("/")[-1]
    new_file_name=os.path.join(best_ods_dir,eval)
    new_file=open(new_file_name,"w")
    model_cntR,model_sumR,model_cntP,model_sumP,model_R,model_P,model_F=[],[],[],[],[],[],[]
    model_count=0
    for model in model_name:
        eval_path=os.path.join(exp_dir+str(model),"nms-eval",eval)
        eval_txt=open(eval_path,"r").readlines()
        
        for index in range(len(eval_txt)):
            str_lines=' '.join(eval_txt[index].split())
            str_lines=str_lines.strip("\n").split(" ")
            cntR,sumR,cntP,sumP=int(str_lines[1]),int(str_lines[2]),int(str_lines[3]),int(str_lines[4])
            model_cntR.append(cntR),model_sumR.append(sumR),model_cntP.append(cntP),model_sumP.append(sumP)

            R,P,F=computeRPF(cntR,sumR,cntP,sumP)
            model_R.append(R),model_P.append(P),model_F.append(F)
   
    model_R,model_p,model_F=np.array(model_R).reshape(len(model_name),100),np.array(model_P).reshape(len(model_name),100),np.array(model_F).reshape(len(model_name),100)
    
    index_selected=np.argmax(model_F,0)#取值范围0-10,各个阈值下选择哪个模型

    img_F=index_select(model_F,index_selected)

    max_F_index=np.argmax(img_F)#选择对于该张图片来说最好的阈值
    
    i_max_loc=np.where(model_F==np.max(model_F))[0].item()
    
    model_cntR, model_sumR, model_cntP, model_sumP=np.array(model_cntR).reshape(len(model_name),100),np.array(model_sumR).reshape(len(model_name),100),np.array(model_cntP).reshape(len(model_name),100),np.array(model_sumP).reshape(len(model_name),100)
    
    i_eval=eval.rindex("_")
    
    record_txt.writelines(eval[:i_eval]+"\t"+str(i_max_loc)+"\n")
    index_model=index_selected[max_F_index]
    oisCntR,oisSumR,oisCntP,oisSumP=oisCntR+model_cntR[index_model][max_F_index],oisSumR+model_sumR[index_model][max_F_index],oisCntP+model_cntP[index_model][max_F_index],oisSumP+model_sumP[index_model][max_F_index]

    img_cntR,img_sumR,img_cntP,img_sumP=index_select(model_cntR,index_selected), index_select(model_sumR,index_selected),index_select(model_cntP,index_selected),index_select(model_sumP,index_selected)

    
    cntR_all,sumR_all,cntP_all,sumP_all=cntR_all+img_cntR,sumR_all+img_sumR,cntP_all+img_cntP,sumP_all+img_sumP
    th = np.arange(0, 1.0, 0.01)
    for i_th in range(100):
        th_c=('%.2f'%th[i_th])
        if i_th==0:
            new_file.writelines("\t\t"+str(int(th[i_th]))+'      '+str(int(img_cntR[i_th].item()))+"      "+str(int(img_sumR[i_th].item()))+"      "+str(int(img_cntP[i_th].item()))+"      "+str(int(img_sumP[i_th].item()))+"\n")
        else:
            new_file.writelines("\t\t"+str(th_c)[:4]+'      '+str(int(img_cntR[i_th].item()))+"      "+str(int(img_sumR[i_th].item()))+"      "+str(int(img_cntP[i_th].item()))+"      "+str(int(img_sumP[i_th].item()))+"\n")
    new_file.flush()
    new_file.close()
R_all,P_all,F_all=computeRPF_numpy(cntR_all,sumR_all,cntP_all,sumP_all)

odsR,odsP,odsF,odsT=findBestRPF(T,R_all,P_all)   
oisR,oisP,oisF=computeRPF(oisCntR,oisSumR,oisCntP,oisSumP)

new_R = np.arange(0.01, 1.01, 0.01)
f = interp1d(R_all[:,0], P_all[:,0], kind='linear')

print('best ODS:'+str(odsF)+',best OIS:'+str(oisF))
record_txt.flush()
record_txt.close()
