#version 410 core

uniform int mm;
out vec4 o;
float _t = (mm - 400)/18900.; // in beats instead of seconds (140 BPM, 44100 sample rate...)
vec2 _res = vec2(640,480);

#define sat(x) clamp(x, 0., 1.)

const float MAX_DIST = 500.;
const float SURF_DIST = .001;
const vec3 FOG_COLOR = vec3(0.569, 0.608, 0.651);
const vec3 SUN_COLOR = vec3(0.937, 0.914, 0.627);
const mat2 m = mat2( 1.6,  1.2, -1.2,  1.6 );
const float PI = 3.141592;

vec2 hash( vec2 p ) {
	p = vec2(dot(p,vec2(127.1,311.7)), dot(p,vec2(269.5,183.3)));
	return -1.0 + 2.0*fract(sin(p)*43758.5453123);
}

float noise( in vec2 p ) {
    const float K1 = 0.366025404; // (sqrt(3)-1)/2;
    const float K2 = 0.211324865; // (3-sqrt(3))/6;
	vec2 i = floor(p + (p.x+p.y)*K1),
        a = p - i + (i.x+i.y)*K2,
        o = (a.x>a.y) ? vec2(1.0,0.0) : vec2(0.0,1.0), //vec2 of = 0.5 + 0.5*vec2(sign(a.x-a.y), sign(a.y-a.x));
        b = a - o + K2,
	    c = a - 1.0 + 2.0*K2;
    vec3 h = max(0.5-vec3(dot(a,a), dot(b,b), dot(c,c) ), 0.0 ),
	     n = h*h*h*h*vec3( dot(a,hash(i+0.0)), dot(b,hash(i+o)), dot(c,hash(i+1.0)));
    return dot(n, vec3(70.0));	
}

float fbm(vec2 n) {
	float total = 0.0, amplitude = 0.1;
	for (int i = 0; i < 7; i++) {
        total += noise(n) * amplitude;
		n = m * n;
		amplitude *= 0.4;
	}
	return total;
}

mat2 rot(float a) {
    float s=sin(a), c=cos(a);
    return mat2(c, -s, s, c);
}

float sdBox(vec3 p, vec3 s) {
    p = abs(p)-s;
	return length(max(p, 0.))+min(max(p.x, max(p.y, p.z)), 0.);
}

float sdHollowCircle(vec2 p, float r, float t) {
    return abs(length(p) - r) - t;
}

float sdCappedCylinder( vec3 p, float h, float r ) {
    vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
    return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdTriPrism( vec3 p, vec2 h ) {
    vec3 q = abs(p);
    return max(q.z-h.y,max(q.x*0.866025+p.y*0.5,-p.y)-h.x*0.5);
}

float opSmoothUnion( float d1, float d2, float k ) {
    float h = sat(0.5 + 0.5*(d2-d1)/k);
    return mix( d2, d1, h ) - k*h*(1.0-h);
}

float tlBaseOutside(vec3 p, float rounding, float thickness) {
    vec3 q = p;
    q.z = -p.z - 0.2;
    
    float d = sdTriPrism(q.xzy, vec2(.7, 1.25));
    d = max(d, -0.3 - p.z);
    
    float r = 30.;
    //d = d + sin(p.x*r)*sin(p.y*r)*sin(p.z*r) * .001; // Displacement
    
    return d;
}

float tlBaseInside(vec3 p) {
    // Box
    float d = sdBox(p, vec3(0.5, 1.1, 0.05));
    
    // Displacement
    float r = 30.;
    p.xy += 0.1;
    
    return d + sin(p.x*r)*sin(p.y*r)*sin(p.z*r) * .0001;
}

float tlLight(vec3 p) {
    // Repeat once towards each direction
    float c = 0.7;
    float l = 1.;
    p.y = p.y-c*clamp(round(p.y/c),-l,l);
    
    return sdCappedCylinder((p - vec3(0.,0.,0.2)).xzy, 0.1, 0.3);
}

float _tlCap(vec3 p) {
    return max(max(sdHollowCircle(p.xy, .34, 0.02), abs(p.z-.4) - .4), -p.y);
}

float tlCaps(vec3 p) {
    vec3 capOffset = vec3(0,0.7,0);
    
    // make 3 caps (finite repetition made glitches which I can't be fucked to solve)
    return max(min(min(_tlCap(p), _tlCap(p - capOffset)), _tlCap(p + capOffset)), length(p.xz - vec2(0., 0.2)) - 0.6);
}

float tlPole(vec3 p) {
    p.y += 2.;
    return sdCappedCylinder(p, 1., 0.1);
}

float floorHeight(vec2 p) {
    // Moves to a local maximum nearby
    vec2 v = vec2(.123, .114);
    
    // the -0.007 is there because for some inexplicable reason, the function returns ~0.007 for (0,0).
    // it shouldn't, and when I set v to be some uniform (with the EXACT same value) it doesn't happen.
    // I can't explain it.
    return fbm(p / 30. + v) - fbm(v) - 0.007;
}

float tlFloor(vec3 p) {
    return p.y + 3. + floorHeight(p.xz) * 60.;
}

vec2 scene(vec3 p) {
     float baseOut = tlBaseOutside(p, 0.3, 0.2),
          baseIn = tlBaseInside(p-vec3(0,0,0.2)),
          pole = tlPole(p),
          light = tlLight(p),
          hole = light - 0.02, // Carve a slightly larger hole from the base
          wholeTL = opSmoothUnion(pole, baseOut, 0.1), // The part of the traffic light that's lighter grey
          caps = tlCaps(p),
          floor_ = tlFloor(p);

    baseIn = max(-hole, baseIn);
    
    float d = min(min(wholeTL, min(baseIn, min(light, caps))), floor_);
    
    float mat = 0.;
    
    if (d == wholeTL) {
        mat = 1.;
    } else if (d == baseIn) {
        mat = 2.;
    } else if (d == light) {
        mat = 3.;
    } else if (d == caps) {
        mat = 4.;
    } else if (d == floor_) {
        mat = 5.;
    }
    
    // *.7 to get rid of shading artifacts
    return vec2(d * .7, mat);
}

vec3 gRayMarch(vec3 ro, vec3 rd, float k) {
    vec3 dO = vec3(0,0,1);

    for(int i=0; i<100; i++) {
    	vec3 p = ro + rd*dO.x;
        vec2 dS = scene(p);
        dO.z = min( dO.z, k*dS.x/dO.x );
        dO.x += dS.x;
        dO.y = dS.y;
        if(dO.x>MAX_DIST || abs(dS.x)<SURF_DIST) break;
    }

    return dO;
}

vec2 rayMarch(vec3 ro, vec3 rd) {
	return gRayMarch(ro,rd,0).xy;
}

float isGloballyLit(vec3 p, vec3 n, vec3 lightDir, float k) {
    return gRayMarch(p + n * SURF_DIST * 2., lightDir, k).z;
}

vec3 getNormal(vec3 p) {
    vec2 e = vec2(.001, 0);
    vec3 n = scene(p).x - 
        vec3(scene(p-e.xyy).x, scene(p-e.yxy).x,scene(p-e.yyx).x);
    
    return normalize(n);
}

vec3 getRayDir(vec2 uv, vec3 p, vec3 l, float z) {
    vec3 
        f = normalize(l-p),
        r = normalize(cross(vec3(0,1,0), f)),
        u = cross(f,r),
        c = f*z,
        i = c + uv.x*r + uv.y*u;
    return normalize(i);
}

vec3 getTLCenter(vec3 p) {
    return vec3(0, round((p.y - 0.1) / 0.7) * 0.7, 0.3);
}

int getTLIndexByPoint(vec3 p) {
    return 1 - int(round(p.y / 0.7));
}

vec3 tlColors[] = vec3[](vec3(1, 0, 0), vec3(1, 1, 0), vec3(0, 1, 0));
vec3 getTLColorByIndex(int i) {
    return tlColors[i];
}

vec3 getTLColor(vec3 p) {
    int i = getTLIndexByPoint(p);
    return getTLColorByIndex(i);
}

vec3 getTLCenterByIndex(int i) {
    return vec3(0, 0.7 * float(1 - i), 0.3);
}

vec3 renderLightMetalMesh(vec3 p) {
    // Renders the "metal mesh" effect often seen in real traffic lights.
    vec3 col = getTLColor(p);
    
    vec2 uv = p.xy * 200.;
    
    // The effect is a bunch of rows, each having tiny bumps throughout, modeled after real traffic lights I've seen
    return col * (sin(uv.x) * .1 + .9) * (sin(uv.y) * .5 + .5);
}

float isLocallyLit(vec3 p, vec3 n, vec3 light, float lightRadius, float k) {
    return sat(dot(normalize(light - p), normalize(n))) * step(abs(p.y - light.y),lightRadius * 1.5);
}

vec3 rgb(int r,int g,int b){return vec3(r,g,b) / 255.;}

vec3 clouds(vec2 uv) {
    float time = _t * (step(62., _t) * step(_t, 80.) * .8 + .1);
    
    vec3 sky = mix(rgb(100,120,141), rgb(204,202,198), sat(uv.y*.1+.2)) * .7;
    
    float cloud1 = sat(fbm(uv+time) * 3.5 + .2);
    float cloud2 = sat(fbm(uv + time * vec2(1.,-1.)) * 3.5 + .2);
    float cloudmix = cloud1 * cloud2 + .12;
    cloudmix = pow(sat(cloudmix * 3.), 2.);
    
    
    /*float cloudmix = 0.;
    
    float w = 0.7;
    for (int i=0; i<8; i++){
		cloudmix += w*noise( uv );
        uv = m*uv + time;
		w *= 0.6;
    }*/
    
    
    
    return mix(sky, vec3(1.), cloudmix);
}

vec3 sky(vec3 rd, vec3 lightDir) {
    float floorY = -5.;
    
    vec2 ti = (rd / clamp(rd.y, 0.001, 1.)).xz; // top intersection
    
    vec3 ceilCol = clouds(ti);//sat(vec3(topIntersect.x, 0., topIntersect.y));
    
    vec3 col = mix(ceilCol, FOG_COLOR, smoothstep(4., 30., length(ti)));

    return mix(col, SUN_COLOR, smoothstep(0.99, 1., sat(dot(rd, lightDir))));
}


vec3 floorTex(vec3 p) {
    vec3 cols[] = vec3[](
        rgb(239,228,227),
        rgb(189,166,160),
        rgb(164,135,111),
        rgb(84,49,16),
        rgb(143,114,81),
        rgb(114,91,64)
        );

    vec2 uv = p.xz / 30.;
    
    vec3 col = mix(cols[2], cols[0], sat(noise(uv / 50.) * 8.));

    float amp = 1.;

    for(int i = 3; i < 6; i++) {
        col = mix(col, cols[i], sat(noise(uv) * amp));
        //amp *= .8;
        uv = m * uv;
    }
    
   /* vec3 scol = clouds(uv);
    scol = vec3(dot(scol, vec3(1./3.)));
    
    col *= (scol * .4 + .6);*/

    col = mix(vec3(.1), col, sat(floorHeight(uv) * 3. + .5));
    
    return col;
}

vec4 render(vec3 p, vec3 rd, vec3 n, vec3 r, float d, float mat, vec3 lightDir) {
    vec3 matColors[] = vec3[](vec3(0.), vec3(.03), vec3(.01), vec3(0.), vec3(.03));
    vec3 suncolor = rgb(239,233,160);

    float tlTime = floor(_t);
    bool isTLLit[] = bool[](tlTime < 224, tlTime == 223, tlTime >= 224);
    
    vec3 col = sky(rd, lightDir);
    float rf = 0.; // reflection

    if(d > MAX_DIST) return vec4(col, rf);
  

    float dif = sat(dot(n, lightDir)*.5+.5);
    float fres = pow(1. - sat(dot(r, normalize(n))), 4.);
    float spec = pow(sat(dot(lightDir, r)), 100.);
    float l = isGloballyLit(p,n,lightDir, 10.);

    if (mat <= 4.) {
        // Shading!
        col = vec3(dif) * matColors[int(mat)] * mix(0.3, 1., l)
            + vec3(spec) * .3 * l * suncolor; // blacken

        // Reflection (except on light and caps)
        if (mat != 3. && mat != 4.)
            rf = .3 * fres;
    } else if (mat == 5.) {
        col = floorTex(p) * pow(dif, 3.) * mix(0.3,1.,l);
    }

    if (mat == 3.) {
        // Compute the traffic light index by Y
        int idx = getTLIndexByPoint(p);
        
        if (isTLLit[idx]) // "Radiant" material
            col = renderLightMetalMesh(p);
    }

    if (mat == 4.) {
        // Compute light from traffic lights
        for (int i = 0; i < 3; i++) {
            if (!isTLLit[i]) continue;
            vec3 light = getTLCenterByIndex(i);
            lightDir = normalize(light - p);

            dif = sat(dot(n, lightDir));
            spec = pow(sat(dot(lightDir, normalize(r))), 2.);

            vec3 tlCol = getTLColorByIndex(i);

            col += tlCol * spec * .5 * isLocallyLit(p,n,light, 0.33, 3.);
        }
    }
    
    // Fog (needs improvement)
    col = mix(col, FOG_COLOR, smoothstep(0., MAX_DIST, d));
    
    return vec4(col, rf);
}

vec3 scene_trafficLight(in vec2 uv, vec2 m, vec3 look, vec3 sunDir) {
    vec3 ro = vec3(0, 4, -4);
    ro.yz *= rot(-m.y*3.141592+1.);
    ro.xz *= rot(-m.x*6.283185);
    ro += look;
    
    vec3 rd = getRayDir(uv, ro, look, .8);
    
    vec2 rm = rayMarch(ro, rd);
    float d = rm.x;
    
    vec3 p = ro + rd * d;

    vec3 n = getNormal(p);
    vec3 r = normalize(reflect(rd, n));
    
    vec3 col = vec3(0);
    
    vec4 renderResult = render(p, rd, n, r, d, rm.y, sunDir);
    col = renderResult.rgb;
    
    float rf = renderResult.a;
    if (rf > 0.) {
        // Reflect
        rd = r;
        ro = p + n * SURF_DIST * 2.;

        rm = rayMarch(ro, rd);

        p = ro + rd * rm.x;
        n = getNormal(p);
        r = normalize(reflect(rd, n));
        
        vec4 renderResult2 = render(p, rd, n, r, rm.x, rm.y, sunDir);
        col += renderResult2.rgb * rf;
    }
    
    return col;
}

float nsin(float a) {
    return (1. - cos(a * PI)) / 2.;
}

vec3 scene_opening(in vec2 uv) {
    float st = mod(_t / 2., 16.);
    st = min(st, st / 2. + 4.);
    
    float fade = nsin(st / 2.);

    vec2 v = uv - 0.3;
    
    float star = pow((sin(atan(v.y,v.x)*6.) + 1.) / 2., 4.);
    float circle = 1. - smoothstep(0., 0.2, length(v));
    float factor = mix(star*circle, circle,circle);
    
    return vec3(1,0,0) * pow(factor, (68. - _t) / 4.) * fade;
}

vec3 animate_trafficLight(vec2 uv) {
    float e = step(_t, 176.), // 1 while not in end
          s = step(80., _t),  // 1 while after start
          f = step(192., _t), // 1 in final shots
          p = smoothstep(160., 222., _t),  // spline or smth
          mo = mod(_t, 8.);

    vec2 m = vec2(.5, .5),
         dir = hash(vec2(_t - mo));

    vec3 look = vec3(1000. * e,0,0), //, 0., _t * -30.); //sin(_t),0.,cos(_t)) * 40.;
         sun = vec3(0,0,-1);

    sun.yz *= rot(smoothstep(64., 80., _t) * 3. - .2);
    sun.xz *= rot(.3);

    look.xz -= mix(smoothstep(0, 32, 191. - _t) * vec2(0, -16), (8. - mo) * dir, e) * 30. * s;
    m += mix(smoothstep(0.,8.,mo)-.5, 1, step(_t, 128.)) * dir * s * e * vec2(.5, .2);
    m += p * f * 30. * vec2(.1,0) * (step(208., _t)*2.-1.);
    look.y += f * (1.-p) * 4.;

    return scene_trafficLight(uv, m, look, sun);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (fragCoord.xy-.5*_res.xy)/_res.y;

    bool isOpening = _t < 62. || (_t < 63. && (uv.x < 0.));
	
    vec3 col = isOpening ? scene_opening(uv) : animate_trafficLight(uv);
    
    col *= 1. - smoothstep(230., 244., _t);

    col = pow(col, vec3(.4545));	// gamma correction
    
    fragColor = vec4(col, 1.0);
}

void main() {
    mainImage(gl_FragColor, gl_FragCoord.xy);
}