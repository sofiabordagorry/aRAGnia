import math, subprocess, os, sys
sys.path.insert(0,'.'); import alpha_lib
from arana import spider
DARK,THREAD,NODE,BLUE="#6B3A18","#C08050","#A2602C","#0B5F88"
SP = spider()

def bicho(cx, cy, width):
    s = width/700.0
    return f'<g transform="translate({cx-500*s:.1f},{cy-465*s:.1f}) scale({s:.4f})">{SP}</g>'

FF = ('<style>@font-face{font-family:"SFR";src:url("file:///System/Library/Fonts/SFNSRounded.ttf");}'
      '@font-face{font-family:"SFCR";src:url("file:///System/Library/Fonts/SFCompactRounded.ttf");}</style>')
FAM = "SFCR"
def wrap(S,body,bg):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{S}" height="{S}" viewBox="0 0 {S} {S}">'
            f'{FF}<rect width="{S}" height="{S}" fill="{bg}"/>{body}</svg>')

def txt(x,y,size,anchor,ls=2):
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{FAM}, Arial Rounded MT Bold, sans-serif" '
            f'font-weight="900" style="font-variation-settings:\'wght\' 900" font-size="{size}" fill="{DARK}" '
            f'letter-spacing="{ls}"><tspan>a</tspan><tspan>RAG</tspan><tspan>nia</tspan></text>')

def mesh(cx, cy, rings, angs, nr, hl, sw, hide_normal=False, closed=False, sag=0.90):
    def pt(a,r): return (cx+r*math.cos(math.radians(a)), cy+r*math.sin(math.radians(a)))
    step = 360.0/len(angs)
    p=[f'<line x1="{cx}" y1="{cy}" x2="{pt(a,rings[-1])[0]:.1f}" y2="{pt(a,rings[-1])[1]:.1f}"/>' for a in angs]
    for r in rings:
        d=[]
        rng = range(len(angs)) if closed else range(len(angs)-1)
        for i in rng:
            a1=angs[i]; a2=angs[(i+1)%len(angs)]
            mid = a1+step/2 if closed else (a1+a2)/2
            x1,y1=pt(a1,r); x2,y2=pt(a2,r); qx,qy=pt(mid, r*sag)
            d.append(f'M {x1:.1f} {y1:.1f} Q {qx:.1f} {qy:.1f} {x2:.1f} {y2:.1f}')
        p.append(f'<path d="{" ".join(d)}"/>')
    n=""
    for i,r in enumerate(rings):
        for j,a in enumerate(angs):
            x,y=pt(a,r); big=(i,j) in hl
            if not big and hide_normal: continue
            n+=f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{nr*2.1 if big else nr:.1f}" fill="{BLUE if big else NODE}"/>'
    return f'<g fill="none" stroke="{THREAD}" stroke-width="{sw}" stroke-linecap="round">{"".join(p)}</g><g>{n}</g>'

A12=[i*30 for i in range(12)]; HL={(3,1),(2,4),(3,6),(2,10)}
FAN=[18+i*(144/8) for i in range(9)]

def horizontal():
    S=1700; CY=850; R=250; mx=520
    return S, mesh(mx,CY,[R*.40,R*.60,R*.80,R],A12,12,HL,6,closed=True)+bicho(mx,CY-20,385)+txt(mx+R+18,CY+62,180,"start")
def vertical():
    S=1200
    return S, mesh(600,490,[130,215,300,372],FAN,11,{(3,1),(2,4),(3,6),(1,7)},6,closed=False,sag=0.86)+bicho(600,305,600)+txt(600,1055,170,"middle")
def icono():
    S=1000; C=500; R=430
    return S, mesh(C,C,[R*.40,R*.60,R*.80,R],A12,17,HL,9,closed=True)+bicho(C,C-35,620)
def favicon():
    S=1000; C=500; R=470; A8=[i*45 for i in range(8)]
    return S, mesh(C,C,[R*.66,R],A8,24,{(1,1),(0,4),(1,6)},30,hide_normal=True,closed=True)+bicho(C,C-10,800)

JOBS=(("aragnia-horizontal",horizontal,2400),("aragnia-logo",vertical,2400),
      ("aragnia-icono",icono,1600),("aragnia-favicon",favicon,1024))
for name,fn,size in JOBS:
    for bg,suf in (("#FFFFFF","_w"),("#000000","_b"),("none","")):
        S,body=fn(); open(f"{name}{suf}.svg","w").write(wrap(S,body,bg))
    for suf in ("_w","_b"):
        f=f"{name}{suf}.svg.png"
        if os.path.exists(f): os.remove(f)
        subprocess.run(["qlmanage","-t","-s",str(size),"-o",".",f"{name}{suf}.svg"],capture_output=True)
    alpha_lib.compose(f"{name}_w.svg.png",f"{name}_b.svg.png",f"{name}.png")
    print(name, alpha_lib.crop(f"{name}.png",16), os.path.getsize(f"{name}.svg"),"bytes svg")
