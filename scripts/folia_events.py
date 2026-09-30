"""Stream a FoLiA XML (Berntzen WhatsApp corpus) from stdin; print actor, datetime, class, text per chat event."""
import html, re, sys

ev = re.compile(r'<event [^>]*actor="([^"]*)"[^>]*begindatetime="([^"]*)"[^>]*class="([^"]*)"')
pending = None
for line in sys.stdin:
    m = ev.search(line)
    if m:
        pending = m.groups()
        continue
    if pending is not None:
        t = re.search(r"<t>(.*)</t>", line)
        txt = html.unescape(t.group(1)) if t else ""
        print(*pending, txt.replace("\t", " "), sep="\t")
        pending = None
