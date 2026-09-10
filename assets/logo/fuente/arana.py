import math
OUT, BODY, LEG, EYE = "#6B3A18", "#EDA96C", "#C8814A", "#4A2410"

def fuzzy(cx, cy, r, n, bump=0.60, phase=0.0):
    pts=[(cx+r*math.cos(2*math.pi*i/n+phase), cy+r*math.sin(2*math.pi*i/n+phase)) for i in range(n)]
    br=2*r*math.sin(math.pi/n)*bump
    d=f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"
    for i in range(1,n+1):
        x,y=pts[i%n]; d+=f" A {br:.1f} {br:.1f} 0 0 1 {x:.1f} {y:.1f}"
    return d+" Z"

LEGS=[((392,452),(292,392),(228,420), 62, 46),
      ((385,516),(262,506),(206,548), 60, 46),
      ((398,574),(296,618),(258,684), 56, 42),
      ((452,598),(430,684),(448,760), 48, 0)]

def spider():
    L=[]; paths=[]; paws=[]
    for a,b,c,w,pw in LEGS:
        paths.append((f'M {a[0]} {a[1]} Q {b[0]} {b[1]} {c[0]} {c[1]}', w))
        am=(1000-a[0],a[1]); bm=(1000-b[0],b[1]); cm=(1000-c[0],c[1])
        paths.append((f'M {am[0]} {am[1]} Q {bm[0]} {bm[1]} {cm[0]} {cm[1]}', w))
        if pw: paws += [(c,pw),(cm,pw)]
    for d,w in paths: L.append(f'<path d="{d}" fill="none" stroke="{OUT}" stroke-width="{w+22}" stroke-linecap="round"/>')
    for (x,y),pw in paws: L.append(f'<path d="{fuzzy(x,y,pw+11,9,0.62)}" fill="{OUT}"/>')
    for d,w in paths: L.append(f'<path d="{d}" fill="none" stroke="{LEG}" stroke-width="{w}" stroke-linecap="round"/>')
    for (x,y),pw in paws: L.append(f'<path d="{fuzzy(x,y,pw,9,0.62)}" fill="{LEG}"/>')
    L.append(f'<path d="{fuzzy(500,392,198,22,0.62)}" fill="{BODY}" stroke="{OUT}" stroke-width="12"/>')
    L.append(f'<path d="{fuzzy(500,512,162,20,0.60,0.14)}" fill="{BODY}" stroke="{OUT}" stroke-width="12"/>')
    for x,y,r in [(374,506,25),(452,524,44),(548,524,44),(626,506,25)]:
        L.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{EYE}"/>')
        L.append(f'<circle cx="{x-r*0.32:.1f}" cy="{y-r*0.36:.1f}" r="{r*0.33:.1f}" fill="#fff"/>')
        if r>30: L.append(f'<circle cx="{x+r*0.32:.1f}" cy="{y+r*0.26:.1f}" r="{r*0.17:.1f}" fill="#fff"/>')
    return "".join(L)

if __name__=="__main__":
    open("arana.svg","w").write(f'<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000" viewBox="0 0 1000 1000"><rect width="1000" height="1000" fill="#fff"/>{spider()}</svg>')
