# Appending `#page=N` to a Windows PDF path makes Edge report "file not found"

**Symptom.** To open a deck at the slide just edited, I launched Edge with a page fragment
glued onto the plain Windows path:

```powershell
Start-Process "msedge.exe" -ArgumentList '"C:\...\ml\12_cnn\L17_cnn_architectures.pdf#page=3"'
Start-Process "msedge.exe" -ArgumentList '"C:\...\ml\12_cnn\L16_cnn_foundations.pdf#page=31"'
```

Both opened Edge on a file-not-found error instead of the deck (instructor, 2026-10-02).
`Start-Process` itself printed nothing, so the shell gave no hint that anything went wrong.

**Cause.** `#page=N` is a URL fragment. On a plain path like `C:\...\deck.pdf#page=3` there is no
URL, so the `#page=3` is read as part of the file name, and no file
`L17_cnn_architectures.pdf#page=3` exists.

**What to do.** Open the plain path, exactly as the project `CLAUDE.md` snippet does, and say the
page number in the message instead:

```powershell
Start-Process "msedge.exe" -ArgumentList '"C:\...\ml\12_cnn\L17_cnn_architectures.pdf"'
```

Untested: a real `file:///C:/.../deck.pdf#page=3` URL might honour the fragment, since it is then a
URL. Check it on one deck before ever relying on it.
