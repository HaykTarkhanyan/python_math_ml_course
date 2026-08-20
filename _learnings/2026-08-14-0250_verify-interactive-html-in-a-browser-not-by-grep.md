# Verify interactive HTML by driving a browser, not by grepping the file

**Symptom.** A self-contained plotly page with 2000 base64 thumbnails embedded in `customdata`
looked broken on inspection. Every check said the images were missing:

```
data:image/jpeg;base64  ->  0 hits
/9j/                    ->  0 hits   (the base64 prefix every JPEG starts with)
image/jpeg              ->  3 hits   (all three inside the plotly bundle itself)
```

**Cause.** Plotly escapes forward slashes as `/` when it serializes the figure to JSON, so
the string in the file is

```
data:image/jpeg;base64,/9j/4AAQSkZJRgABAQ...
```

which is valid JSON, decodes back to a correct data URI in the browser, and matches none of the
obvious greps. Nothing was wrong. The verification method was.

**What actually settles it.** Serve the file and drive a real browser (Playwright's `file:`
protocol is blocked, so `python -m http.server` in the output directory, then navigate to
`127.0.0.1`). Dispatch a real `mousemove` at a point's pixel position and read back the tooltip
state:

```js
{ boxDisplay: "block", imgSrcPrefix: "data:image/jpeg;base64,/9j/4AAQS", imgLen: 1611 }
```

That is proof. A string search over a generated file is not.

**Second trap in the same check.** With plotly 6, `gd.data[0].x` is the *serialized* form and may
be a base64-packed object rather than an array, so `gd.data[0].x[0]` is `undefined` and any pixel
position computed from it comes out `NaN`. The expanded values live on **`gd._fullData`**. Two
attempts at the hover test failed with "non-finite clientX" before this was the culprit.

**Consequences / rule.** For any generated interactive artifact - HTML reports, plotly pages,
anything with embedded assets or JS - the definition of done is *it was opened and exercised*, not
*the expected substring is present*. Grep can only prove a file contains what you guessed it would
contain, and encoders are free to encode.

Built during the ch4 clustering photo-grouping practical; see
`ml/09_clustering/33_image_clusters_solution.ipynb`.
