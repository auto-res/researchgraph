(()=>{"use strict";var e,a,c,t,r,f={},d={};function o(e){var a=d[e];if(void 0!==a)return a.exports;var c=d[e]={id:e,loaded:!1,exports:{}};return f[e].call(c.exports,c,c.exports,o),c.loaded=!0,c.exports}o.m=f,o.c=d,e=[],o.O=(a,c,t,r)=>{if(!c){var f=1/0;for(i=0;i<e.length;i++){c=e[i][0],t=e[i][1],r=e[i][2];for(var d=!0,b=0;b<c.length;b++)(!1&r||f>=r)&&Object.keys(o.O).every((e=>o.O[e](c[b])))?c.splice(b--,1):(d=!1,r<f&&(f=r));if(d){e.splice(i--,1);var n=t();void 0!==n&&(a=n)}}return a}r=r||0;for(var i=e.length;i>0&&e[i-1][2]>r;i--)e[i]=e[i-1];e[i]=[c,t,r]},o.n=e=>{var a=e&&e.__esModule?()=>e.default:()=>e;return o.d(a,{a:a}),a},c=Object.getPrototypeOf?e=>Object.getPrototypeOf(e):e=>e.__proto__,o.t=function(e,t){if(1&t&&(e=this(e)),8&t)return e;if("object"==typeof e&&e){if(4&t&&e.__esModule)return e;if(16&t&&"function"==typeof e.then)return e}var r=Object.create(null);o.r(r);var f={};a=a||[null,c({}),c([]),c(c)];for(var d=2&t&&e;"object"==typeof d&&!~a.indexOf(d);d=c(d))Object.getOwnPropertyNames(d).forEach((a=>f[a]=()=>e[a]));return f.default=()=>e,o.d(r,f),r},o.d=(e,a)=>{for(var c in a)o.o(a,c)&&!o.o(e,c)&&Object.defineProperty(e,c,{enumerable:!0,get:a[c]})},o.f={},o.e=e=>Promise.all(Object.keys(o.f).reduce(((a,c)=>(o.f[c](e,a),a)),[])),o.u=e=>"assets/js/"+({155:"c864d414",547:"f64af1b0",639:"3640b643",867:"33fc5bb8",1235:"a7456010",1246:"1a18a058",1724:"dff1c289",1903:"acecf23e",1953:"1e4232ab",1972:"73664a40",1974:"5c868d36",2104:"4ae3ac19",2172:"511f5cc9",2485:"d5c52e6f",2711:"9e4087bc",2748:"822bd8ab",2824:"052f5ed8",3098:"533a09ca",3249:"ccc49370",3252:"4c3bedd0",3637:"f4f34a3a",3694:"8717b14a",3859:"1af2b7e0",3976:"0e384e19",4134:"393be207",4212:"621db11d",4301:"4fceb789",4583:"1df93b7f",4736:"e44a2883",4813:"6875c492",5557:"d9f32620",5742:"aba21aa0",6061:"1f391b9e",6289:"d74e223e",6969:"14eb3368",7098:"a7bd4aaa",7472:"814f3328",7643:"a6aa9e1f",8209:"01a85c17",8401:"17896441",8609:"925b3f96",8737:"7661071f",8863:"f55d3e7a",9048:"a94703ab",9122:"c0930264",9262:"18c41134",9325:"59362658",9328:"e273c56f",9647:"5e95c892",9858:"36994c47"}[e]||e)+"."+{155:"85a5806f",547:"2ad90fa1",639:"738827e7",867:"fdcc920b",1235:"5f9bbb01",1246:"6f502299",1724:"4d8d6985",1903:"cac481dc",1953:"f086b370",1972:"2bc8e09e",1974:"a5de8fb6",2104:"d75ebb51",2172:"8447a1c1",2485:"45b02603",2711:"b4c318cd",2748:"9028fc4a",2824:"d1b88ed0",3042:"c46c6bc5",3098:"519768fa",3249:"126bece0",3252:"009a4ab3",3637:"c6c95a57",3694:"a0b5a425",3859:"35712961",3976:"0a707fe3",4134:"46a4e5c8",4212:"5888e1e9",4301:"3b7a7fd7",4583:"d750b725",4622:"b0619580",4736:"28876944",4813:"5456629d",5557:"2f6960d4",5742:"ed09cce9",6061:"40a83c6c",6289:"a85c2a38",6969:"b93a9a2d",7098:"9373de31",7472:"9b10361e",7643:"b0abcfbd",8209:"ba7daae7",8401:"48533033",8609:"5fd370df",8737:"ab147f89",8863:"cb28411a",9048:"be591cd2",9122:"4d54ee51",9262:"80740255",9325:"7090b8f1",9328:"904ee187",9392:"6bcc3182",9647:"8f639fe6",9858:"337a7516"}[e]+".js",o.miniCssF=e=>{},o.g=function(){if("object"==typeof globalThis)return globalThis;try{return this||new Function("return this")()}catch(e){if("object"==typeof window)return window}}(),o.o=(e,a)=>Object.prototype.hasOwnProperty.call(e,a),t={},r="docs:",o.l=(e,a,c,f)=>{if(t[e])t[e].push(a);else{var d,b;if(void 0!==c)for(var n=document.getElementsByTagName("script"),i=0;i<n.length;i++){var u=n[i];if(u.getAttribute("src")==e||u.getAttribute("data-webpack")==r+c){d=u;break}}d||(b=!0,(d=document.createElement("script")).charset="utf-8",d.timeout=120,o.nc&&d.setAttribute("nonce",o.nc),d.setAttribute("data-webpack",r+c),d.src=e),t[e]=[a];var l=(a,c)=>{d.onerror=d.onload=null,clearTimeout(s);var r=t[e];if(delete t[e],d.parentNode&&d.parentNode.removeChild(d),r&&r.forEach((e=>e(c))),a)return a(c)},s=setTimeout(l.bind(null,void 0,{type:"timeout",target:d}),12e4);d.onerror=l.bind(null,d.onerror),d.onload=l.bind(null,d.onload),b&&document.head.appendChild(d)}},o.r=e=>{"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})},o.p="/researchgraph/",o.gca=function(e){return e={17896441:"8401",59362658:"9325",c864d414:"155",f64af1b0:"547","3640b643":"639","33fc5bb8":"867",a7456010:"1235","1a18a058":"1246",dff1c289:"1724",acecf23e:"1903","1e4232ab":"1953","73664a40":"1972","5c868d36":"1974","4ae3ac19":"2104","511f5cc9":"2172",d5c52e6f:"2485","9e4087bc":"2711","822bd8ab":"2748","052f5ed8":"2824","533a09ca":"3098",ccc49370:"3249","4c3bedd0":"3252",f4f34a3a:"3637","8717b14a":"3694","1af2b7e0":"3859","0e384e19":"3976","393be207":"4134","621db11d":"4212","4fceb789":"4301","1df93b7f":"4583",e44a2883:"4736","6875c492":"4813",d9f32620:"5557",aba21aa0:"5742","1f391b9e":"6061",d74e223e:"6289","14eb3368":"6969",a7bd4aaa:"7098","814f3328":"7472",a6aa9e1f:"7643","01a85c17":"8209","925b3f96":"8609","7661071f":"8737",f55d3e7a:"8863",a94703ab:"9048",c0930264:"9122","18c41134":"9262",e273c56f:"9328","5e95c892":"9647","36994c47":"9858"}[e]||e,o.p+o.u(e)},(()=>{var e={5354:0,1869:0};o.f.j=(a,c)=>{var t=o.o(e,a)?e[a]:void 0;if(0!==t)if(t)c.push(t[2]);else if(/^(1869|5354)$/.test(a))e[a]=0;else{var r=new Promise(((c,r)=>t=e[a]=[c,r]));c.push(t[2]=r);var f=o.p+o.u(a),d=new Error;o.l(f,(c=>{if(o.o(e,a)&&(0!==(t=e[a])&&(e[a]=void 0),t)){var r=c&&("load"===c.type?"missing":c.type),f=c&&c.target&&c.target.src;d.message="Loading chunk "+a+" failed.\n("+r+": "+f+")",d.name="ChunkLoadError",d.type=r,d.request=f,t[1](d)}}),"chunk-"+a,a)}},o.O.j=a=>0===e[a];var a=(a,c)=>{var t,r,f=c[0],d=c[1],b=c[2],n=0;if(f.some((a=>0!==e[a]))){for(t in d)o.o(d,t)&&(o.m[t]=d[t]);if(b)var i=b(o)}for(a&&a(c);n<f.length;n++)r=f[n],o.o(e,r)&&e[r]&&e[r][0](),e[r]=0;return o.O(i)},c=self.webpackChunkdocs=self.webpackChunkdocs||[];c.forEach(a.bind(null,0)),c.push=a.bind(null,c.push.bind(c))})()})();