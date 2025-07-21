find references of the original paper.

## List the References
read the reference list `analysis\references\source-ref.md`, and extract the references used in the paper. List them in a structured format:

- [year][published_where][paper link][github repo link] **title**, first author

Note that:
- references should be sorted by year
- paper titles should be bolded using markdown **bold** formatting

- the paper link should be:
- - if arxiv html link is available, use it, and the link text is `arxiv`
- - otherwise, use the DOI link, link text is `doi`
- - otherwise use the publisher link, link text is `pub`
- - otherwise, just says [NO LINK]
- - if any of the links are available, the link should be clickable

- the github repo link text is `code`, and the link should be clickable if available, otherwise just says `NO CODE`

## Known Issues

### Markdown Bracket Display Issue
When creating clickable links in markdown that should display brackets, need to escape the brackets in the link text:
- **Correct**: `[\[arxiv\]](url)` → displays as clickable `[arxiv]`
- **Incorrect**: `[arxiv](url)` → displays as clickable `arxiv` (no brackets)

This ensures all reference format brackets `[year][venue][link][code]` are visible in the final rendered markdown.