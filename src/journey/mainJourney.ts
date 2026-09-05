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


// ===
// this section I generated with gemini
// ===
const modules = import.meta.glob('./markdowns/*.md', { query: '?raw', import: 'default' });

for (const [path, loader] of Object.entries(modules)) {

  const name = path.split('/').pop()?.replace('.md', '');
  const el = document.getElementById(`markdown-${name}`);

  if (el) {
    const rawText = (await loader()) as string;
    el.innerHTML = await marked.parse(rawText);
  }
}
// ===