"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[7131],{8453:(e,n,r)=>{r.d(n,{R:()=>i,x:()=>s});var t=r(6540);const o={},c=t.createContext(o);function i(e){const n=t.useContext(c);return t.useMemo((function(){return"function"==typeof e?e(n):{...n,...e}}),[n,e])}function s(e){let n;return n=e.disableParentContext?"function"==typeof e.components?e.components(o):e.components||o:i(e.components),t.createElement(c.Provider,{value:n},e.children)}},8939:(e,n,r)=>{r.r(n),r.d(n,{assets:()=>a,contentTitle:()=>s,default:()=>l,frontMatter:()=>i,metadata:()=>t,toc:()=>u});const t=JSON.parse('{"id":"component/executor","title":"Executor","description":"\u3053\u306e\u30da\u30fc\u30b8\u3067\u306f Executor \u30b5\u30d6\u30b0\u30e9\u30d5\u306e\u8a73\u7d30\u306b\u3064\u3044\u3066\u8aac\u660e\u3057\u307e\u3059\u3002","source":"@site/i18n/ja/docusaurus-plugin-content-docs/current/component/executor.md","sourceDirName":"component","slug":"/component/executor","permalink":"/researchgraph/ja/docs/component/executor","draft":false,"unlisted":false,"editUrl":"https://github.com/auto-res/researchgraph/tree/main/docs/docs/component/executor.md","tags":[],"version":"current","sidebarPosition":2,"frontMatter":{"id":"executor","title":"Executor","sidebar_position":2},"sidebar":"tutorialSidebar","previous":{"title":"Generator","permalink":"/researchgraph/ja/docs/component/generator"},"next":{"title":"HTML uploader","permalink":"/researchgraph/ja/docs/component/html-uploader"}}');var o=r(4848),c=r(8453);const i={id:"executor",title:"Executor",sidebar_position:2},s="Executor \u30b5\u30d6\u30b0\u30e9\u30d5",a={},u=[{value:"\u6982\u8981",id:"\u6982\u8981",level:2},{value:"\u4e3b\u306a\u6a5f\u80fd",id:"\u4e3b\u306a\u6a5f\u80fd",level:2},{value:"\u4f7f\u3044\u65b9",id:"\u4f7f\u3044\u65b9",level:2},{value:"API",id:"api",level:2}];function d(e){const n={code:"code",h1:"h1",h2:"h2",header:"header",li:"li",p:"p",pre:"pre",ul:"ul",...(0,c.R)(),...e.components};return(0,o.jsxs)(o.Fragment,{children:[(0,o.jsx)(n.header,{children:(0,o.jsx)(n.h1,{id:"executor-\u30b5\u30d6\u30b0\u30e9\u30d5",children:"Executor \u30b5\u30d6\u30b0\u30e9\u30d5"})}),"\n",(0,o.jsx)(n.p,{children:"\u3053\u306e\u30da\u30fc\u30b8\u3067\u306f Executor \u30b5\u30d6\u30b0\u30e9\u30d5\u306e\u8a73\u7d30\u306b\u3064\u3044\u3066\u8aac\u660e\u3057\u307e\u3059\u3002"}),"\n",(0,o.jsx)(n.h2,{id:"\u6982\u8981",children:"\u6982\u8981"}),"\n",(0,o.jsx)(n.p,{children:"Executor \u30b5\u30d6\u30b0\u30e9\u30d5\u306f\u3001\u8ad6\u6587\u306e\u30b3\u30fc\u30c9\u5b9f\u884c\u3084\u5b9f\u9a13\u306e\u5b9f\u884c\u3092\u62c5\u3046\u30b3\u30f3\u30dd\u30fc\u30cd\u30f3\u30c8\u3067\u3059\u3002"}),"\n",(0,o.jsx)(n.h2,{id:"\u4e3b\u306a\u6a5f\u80fd",children:"\u4e3b\u306a\u6a5f\u80fd"}),"\n",(0,o.jsxs)(n.ul,{children:["\n",(0,o.jsx)(n.li,{children:"\u4e3b\u306a\u6a5f\u80fd1"}),"\n",(0,o.jsx)(n.li,{children:"\u4e3b\u306a\u6a5f\u80fd2"}),"\n",(0,o.jsx)(n.li,{children:"\u4e3b\u306a\u6a5f\u80fd3"}),"\n"]}),"\n",(0,o.jsx)(n.h2,{id:"\u4f7f\u3044\u65b9",children:"\u4f7f\u3044\u65b9"}),"\n",(0,o.jsx)(n.pre,{children:(0,o.jsx)(n.code,{className:"language-python",children:'from researchgraph.executor_subgraph.executor_subgraph import Executor\n\nmax_code_fix_iteration = 3\n\nexecutor = Executor(\n    github_repository=github_repository,\n    branch_name=branch_name,\n    save_dir=save_dir,\n    max_code_fix_iteration=max_code_fix_iteration,\n)\n\nresult = executor.run()\nprint(f"result: {result}")\n'})}),"\n",(0,o.jsx)(n.h2,{id:"api",children:"API"}),"\n",(0,o.jsx)(n.p,{children:"API\u306e\u8a73\u7d30\u306f\u6e96\u5099\u4e2d\u3067\u3059\u3002"})]})}function l(e={}){const{wrapper:n}={...(0,c.R)(),...e.components};return n?(0,o.jsx)(n,{...e,children:(0,o.jsx)(d,{...e})}):d(e)}}}]);