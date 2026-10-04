"""Publication-style topology legend, model comparisons, and recorded 3-D replays."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.animation import FFMpegWriter
from experiment import BASE

BG='#101725';PANEL='#182337';FG='#f0f4fa';MUTED='#aabbd0';BLUE='#6fb5ff';GREEN='#6ee1bd';ORANGE='#ffbd78'
MEMS=['ff','gru','reservoir','spiking'];ENCS=['xyz','invariant']
NAMES={'ff':'Feed-forward','gru':'GRU recurrent','reservoir':'Fixed reservoir','spiking':'Spiking LIF'}
IDS={f'{e}_{m}':f'{"C" if e=="xyz" else "G"}-{dict(ff="FF",gru="GRU",reservoir="ESN",spiking="SNN")[m]}' for e in ENCS for m in MEMS}
MODELS=['classical']+list(IDS)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':FG,
        'axes.labelcolor':MUTED,'axes.edgecolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
        'savefig.facecolor':BG})


def box(ax,x,y,w,h,text,color=BLUE,fs=10):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=.02',
                              edgecolor=color,facecolor=PANEL,lw=1.3))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs)


def arrow(ax,a,b,color=MUTED,style='-'):
    ax.annotate('',b,a,arrowprops=dict(arrowstyle='->',color=color,lw=1.3,linestyle=style))


def topology(training,out):
    style();rows=json.loads((training/'training.json').read_text())
    fig,axes=plt.subplots(4,2,figsize=(18,14),facecolor=BG)
    for row,kind in enumerate(MEMS):
        for col,enc in enumerate(ENCS):
            ax=axes[row,col];ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
            item=next(r for r in rows if r['encoder']==enc and r['memory']==kind);name=IDS[f'{enc}_{kind}'];color=BLUE if enc=='xyz' else GREEN
            ax.text(.01,.95,f'{name}  |  {NAMES[kind]}',fontsize=16,weight='bold',color=color)
            geometry='Raw XYZ + candidate direction' if enc=='xyz' else 'Distances + dot products (rigid invariants)'
            ax.text(.01,.85,geometry,fontsize=11,color=MUTED)
            box(ax,.01,.49,.20,.23,f'64 point slots\nshared point MLP\n6 → {item["encoder_width"]} → 8',color)
            box(ax,.28,.49,.20,.23,'Masked mean + max\n16 pooled features\n+ 8 cue/motion values',color,9)
            box(ax,.55,.49,.14,.23,'Embedding\n24 → 16\ntanh',color)
            arrow(ax,(.215,.605),(.275,.605));arrow(ax,(.485,.605),(.545,.605))
            core={'ff':'MLP 16 → 16\nno temporal state','gru':'GRU: 16 units\nupdate / reset gates',
                  'reservoir':'16 leaky units\nfixed input + recurrence','spiking':'8 LIF neurons\nvoltage + prior spikes\n4 substeps / observation'}[kind]
            box(ax,.75,.47,.23,.27,core,color,10)
            arrow(ax,(.695,.605),(.745,.605))
            if kind!='ff':
                ax.annotate('',(.90,.755),(.79,.755),arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=-.9',
                            color=color,linestyle='--' if kind=='reservoir' else '-'))
            box(ax,.55,.10,.43,.22,'Direction head: embedding + core → scalar\nOccupancy head: embedding → logit',color,10)
            arrow(ax,(.87,.46),(.87,.33));arrow(ax,(.62,.48),(.62,.33))
            fixed=f'; {item["fixed_parameters"]} fixed' if kind=='reservoir' else ''
            ax.text(.01,.33,f'{item["parameters"]:,} weights{fixed}',fontsize=11,color=color)
            ax.text(.01,.21,f'{item["persistent_state_floats"]} temporal floats total\n{item["dense_macs"]/1e6:.3f} M dense MACs / observation',fontsize=10,color=MUTED)
    fig.suptitle('Network topology legend: 2 perception encoders × 4 memory cores',fontsize=23,weight='bold',y=.982)
    fig.text(.06,.942,'Each network shares weights across six candidate directions. The planner uses the six predicted direction and occupancy scores.',color=MUTED,fontsize=12)
    fig.subplots_adjust(top=.905,bottom=.09,left=.055,right=.975,hspace=.10,wspace=.12)
    fig.text(.06,.054,'C = conventional coordinates     G = geometric invariants     Dashed recurrence = fixed reservoir weights',fontsize=12)
    fig.text(.06,.024,'Common to every policy: occupancy mapping, search history, pathfinding, listening schedule, and return control. FF is not an entirely memoryless robot.',fontsize=11,color=MUTED)
    for suffix in ['png','svg']:fig.savefig(out/f'network_topologies.{suffix}',dpi=150)
    plt.close(fig)


def comparisons(evaluation,out):
    style();rows=json.loads((evaluation/'summary.json').read_text());audits=json.loads((evaluation/'audits.json').read_text())
    profiles=['nominal','dropout','multipath','drift','combined','rotated']
    fig,axes=plt.subplots(1,2,figsize=(17,8),facecolor=BG)
    for ax,metric,title in [(axes[0],'success','Complete mission success'),(axes[1],'targets','Targets actually located')]:
        values=[];labels=[]
        for model in MODELS:
            vals=[];labs=[]
            for profile in profiles:
                r=next(r for r in rows if r['model']==model and r['profile']==profile)
                numerator=r['successes'] if metric=='success' else r['targets_found']
                denominator=r['episodes'] if metric=='success' else r['targets_total']
                vals.append(100*numerator/denominator);labs.append(f'{numerator}/{denominator}')
            values.append(vals);labels.append(labs)
        ax.imshow(values,vmin=0,vmax=100,cmap='viridis',aspect='auto')
        ax.set_xticks(range(6),['Nominal','Missing\nreadings','False\nreturns','Pose\ndrift','Combined','Rotated\ncoordinates'],fontsize=10)
        ax.set_yticks(range(9),['Classical']+[IDS[m] for m in MODELS[1:]])
        ax.set_title(title,fontsize=16,pad=14)
        for y in range(9):
            for x in range(6):ax.text(x,y,labels[y][x],ha='center',va='center',color='black' if values[y][x]>65 else 'white',fontsize=10)
    fig.suptitle('3-D search benchmark: all eight learned topologies and the classical control',fontsize=21,weight='bold',y=.98)
    fig.subplots_adjust(top=.89,bottom=.17,left=.09,right=.98,wspace=.28)
    fig.text(.06,.086,'Learned models: 8 shared scenes × 3 training seeds per condition. Classical: the same 8 scenes, with no training seed.',color=MUTED,fontsize=11)
    fig.text(.06,.045,'Success requires all three real targets AND return to base. False detections cannot earn success. Each mission allows 360 actions.',color=MUTED,fontsize=11)
    for suffix in ['png','svg']:fig.savefig(out/f'simulation_comparison.{suffix}',dpi=150)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(17,5.8),facecolor=BG)
    labels=[IDS[m] for m in MODELS[1:]];x=np.arange(8);colors=[BLUE]*4+[GREEN]*4
    for ax in axes:
        ax.set_facecolor(PANEL);ax.set_xticks(x,labels,rotation=45,ha='right');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.12)
    delayed=[];reset=[];accuracy=[];error=[]
    for model in MODELS[1:]:
        group=[a for a in audits if a['model']==model]
        delayed.append(np.mean([a['delayed_target_mse'] for a in group]));reset.append(np.mean([a['reset_target_mse'] for a in group]))
        accuracy.append(np.mean([a['occupancy_balanced_accuracy'] for a in group]));error.append(np.mean([a['rotation_max_abs_logit_difference'] for a in group]))
    axes[0].bar(x-.18,delayed,width=.36,color=colors,label='Normal state')
    axes[0].bar(x+.18,reset,width=.36,color='#d8a07d',label='Reset each observation');axes[0].set_title('Direction error after a cue disappears');axes[0].set_ylabel('Mean squared error; lower is better');axes[0].legend(frameon=False,fontsize=9)
    axes[1].bar(x,np.array(accuracy)*100,color=colors);axes[1].set_ylim(0,100);axes[1].set_title('Local occupancy perception');axes[1].set_ylabel('Balanced accuracy / %')
    axes[2].bar(x,np.maximum(error,1e-8),color=colors);axes[2].set_yscale('log');axes[2].set_title('Coordinate-rotation sensitivity');axes[2].set_ylabel('Mean of per-seed maximum logit change')
    fig.suptitle('Shared held-out sensor traces: geometry and memory examined separately',fontsize=20,weight='bold',y=.99)
    fig.subplots_adjust(top=.84,bottom=.28,left=.065,right=.98,wspace=.34)
    fig.text(.06,.045,'Same observations for every model; direction errors use 2–15 steps after the last cue. Mean over three training seeds.',fontsize=11,color=MUTED)
    fig.savefig(out/'perception_memory_audit.png',dpi=150);plt.close(fig)


def replay_grid(evaluation,out):
    style();replays={m:json.loads((evaluation/f'replay_{m}.json').read_text()) for m in MODELS[1:]}
    first=next(iter(replays.values()))['replay'];blocked=np.array(first['blocked']);targets=np.array(first['targets'])
    fig=plt.figure(figsize=(18,10.5),facecolor=BG);axes=[];paths=[];dots=[];clouds=[];stats=[];marks=[]
    fig.text(.035,.955,'THE SAME 3-D SCENE, EIGHT LEARNED CONTROLLERS',fontsize=22,weight='bold')
    fig.text(.035,.921,'Fixed scene 42, training seed 11. Full geometry is visible to the observer; controllers receive partial sensor returns.',color=MUTED,fontsize=12)
    for i,(name,data) in enumerate(replays.items()):
        ax=fig.add_subplot(2,4,i+1,projection='3d');axes.append(ax);ax.set_facecolor(BG)
        # Show internal walls/floor sparsely, leaving interiors readable.
        interior=blocked[(blocked>0).all(1)&(blocked<np.array(first['shape'])-1).all(1)]
        ax.scatter(*interior.T,s=9,c='#7c91a9',alpha=.13,marker='s',depthshade=False)
        marks.append(ax.scatter(*targets.T,c='#ff788c',s=42,marker='X',depthshade=False))
        paths.append(ax.plot([],[],[],c=BLUE if name.startswith('xyz') else GREEN,lw=2)[0])
        dots.append(ax.scatter([],[],[],c=ORANGE,s=50,depthshade=False))
        clouds.append(ax.scatter([],[],[],c='white',s=4,alpha=.35,depthshade=False))
        ax.set(xlim=(0,12),ylim=(0,12),zlim=(0,8));ax.view_init(elev=25,azim=-55)
        ax.set_xticks([0,6,12],['0','3','6']);ax.set_yticks([0,6,12],['0','3','6']);ax.set_zticks([0,4,8],['0','2','4'])
        ax.tick_params(labelsize=8,pad=0);ax.set_xlabel('x / m',labelpad=-3,fontsize=8);ax.set_ylabel('y / m',labelpad=-3,fontsize=8)
        ax.set_title(f'{IDS[name]}',fontsize=14,weight='bold',color=BLUE if name.startswith('xyz') else GREEN,pad=-7)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False;axis.line.set_color(MUTED)
        stats.append(ax.text2D(.02,-.04,'',transform=ax.transAxes,color=FG,fontsize=10))
    fig.subplots_adjust(top=.87,bottom=.105,left=.015,right=.995,hspace=.15,wspace=.0)
    time_label=fig.text(.97,.955,'',ha='right',fontsize=14)
    fig.text(.035,.055,'White dots: current 3-D range returns   •   colored line: executed path   •   red/green X: hidden/reported real target',fontsize=11,color=MUTED)
    fig.text(.035,.020,'Point-agent voxel kinematics; synthetic sensing; no flight dynamics, debris mechanics, physical extraction, or hardware energy measurement.',fontsize=10,color=MUTED)
    def draw(t):
        time_label.set_text(f't = {t} simulated s')
        for i,(name,data) in enumerate(replays.items()):
            rep=data['replay'];frame=max((f for f in rep['frames'] if f['t']<=t),key=lambda f:f['t'])
            trail=np.array(rep['trail']);trail=trail[:min(t+1,len(trail))]
            paths[i].set_data_3d(trail[:,0],trail[:,1],trail[:,2]);p=trail[-1];dots[i]._offsets3d=tuple(np.array([n]) for n in p)
            pc=np.array(frame['points']).reshape(-1,3);clouds[i]._offsets3d=tuple(pc[:,j] for j in range(3))
            found=frame['found'] if t<data['ticks'] else list(map(int,data['discovery']))
            marks[i].set_color([GREEN if j in found else '#ff788c' for j in range(3)])
            if t>=data['ticks']:text=f"Ended: {data['targets_found']}/3 found | home: {'yes' if data['returned'] else 'no'}"
            else:text=f"{len(found)}/3 found | budget left {frame['energy']} | {frame['mode']}"
            stats[i].set_text(text)
    ffmpeg=shutil.which('ffmpeg') or r'C:\Program Files\LAMMPS2023\bin\ffmpeg.exe'
    plt.rcParams['animation.ffmpeg_path']=ffmpeg
    writer=FFMpegWriter(fps=10,codec='libx264',bitrate=4500,extra_args=['-pix_fmt','yuv420p','-movflags','+faststart'])
    end=max(d['ticks'] for d in replays.values())
    with writer.saving(fig,str(out/'eight_models_3d.mp4'),dpi=100):
        for t in range(0,end+4,4):
            draw(min(t,end))
            for _ in range(2):writer.grab_frame()
        for _ in range(25):writer.grab_frame()
    draw(end);fig.savefig(out/'eight_models_3d.png',dpi=150);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--training',type=Path,default=BASE/'outputs/training_v2')
    p.add_argument('--evaluation',type=Path,default=BASE/'outputs/evaluation');p.add_argument('--out',type=Path,default=BASE/'outputs/figures')
    p.add_argument('--topology-only',action='store_true');args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    topology(args.training,args.out)
    if not args.topology_only:comparisons(args.evaluation,args.out);replay_grid(args.evaluation,args.out)
    print('Rendered',args.out,flush=True)


if __name__=='__main__':main()
