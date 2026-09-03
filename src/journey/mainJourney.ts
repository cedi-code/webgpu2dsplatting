import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';
import { createHighlighter } from 'shiki';
import markedShiki from 'marked-shiki';
import {
  transformerNotationDiff,
  transformerNotationHighlight,
  transformerNotationWordHighlight,
  transformerNotationFocus,
  transformerNotationErrorLevel,
  transformerMetaHighlight,
  transformerMetaWordHighlight
} from '@shikijs/transformers'

// Initialize Shiki
const highlighter = await createHighlighter({
  themes: ['github-dark-dimmed'],
  langs: ['typescript', 'javascript', 'html', 'css', 'glsl', 'wgsl', `markdown`]
});

// init marked (markdown parser, copied from docs: https://www.npmjs.com/package/marked-shiki)
marked
.use(
    markedKatex({
      throwOnError: false,
      nonStandard: true,
    })
  )
.use(markedShiki({
  highlight(code, lang, props) {
    return highlighter.codeToHtml(code, {
      lang,
      theme: 'github-dark-dimmed',
      meta: { __raw: props.join(' ') }, // required by `transformerMeta*`
      transformers: [
        transformerNotationDiff({
          matchAlgorithm: 'v3'
        }),
        transformerNotationHighlight({
          matchAlgorithm: 'v3'
        }),
        transformerNotationWordHighlight({
          matchAlgorithm: 'v3'
        }),
        transformerNotationFocus({
          matchAlgorithm: 'v3'
        }),
        transformerNotationErrorLevel({
          matchAlgorithm: 'v3'
        }),
        transformerMetaHighlight(),
        transformerMetaWordHighlight()
      ]
    })
  }
}
)); 

interface MarkdownSection {
  id: string;
  md: string;
}

const mdPath: string = "src/journey/markdowns/"

const markdownSections: MarkdownSection[] = [
  { 
    id: 'markdown-intro', 
    md: 'intro.md' 
},
  { 
    id: 'markdown-setup', 
    md: 'setup.md' 
},
  { 
    id: 'markdown-drawgauss',
    md: 'splat.md' 
}, 
  {
    id: 'markdown-elipsoid',
    md: 'splatEdit.md'
}
];

// Sequential Load & Render
for (const { id, md } of markdownSections) {
  const el = document.getElementById(id);
  if (!el) continue;

  const response = await fetch(mdPath + md);
  const rawText = await response.text();
  el.innerHTML = await marked.parse(rawText);
}
