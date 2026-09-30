"""c2 — 15 cenários roteirizados (20–40 turnos). O ROTEIRO (quem fala, quando, o que acontece e as marcas #tag) é escrito
à mão aqui; o deepseek-flash só redige cada linha como mensagem de chat realista (sem mudar o conteúdo).
Os critérios de plausibilidade de cada cenário ficam em c2_scen_run.py e foram escritos ANTES de rodar a cascata.
Saída: analysis/data/c2_scenarios.json"""
import json, re
from concurrent.futures import ThreadPoolExecutor
from c2_common import llm, parse_json, ADATA, jdump

C, U = "Mia", "Leo"
# cada linha: "d<dia> HH:MM L|M gist [#tag]"
SCEN = {
 "S01_sincere_apology": ("Close friends with a crush. Leo cancels dinner rudely, next day apologizes sincerely and reschedules.", """
d1 18:00 L hey, still on for dinner at 8?
d1 18:01 M yes!! booked the thai place
d1 18:03 L nice
d1 18:05 M been looking forward to it all week
d1 19:40 L actually can't come, going out with the guys instead, it's whatever #cancel
d1 19:41 M wait what. i'm literally ready to leave
d1 19:43 L relax it's just dinner #dismiss
d1 19:44 M wow ok
d1 19:50 M enjoy your night then
d2 10:15 L hey #after
d2 10:40 M hi
d2 10:42 L i was a jerk yesterday. i'm really sorry, you'd booked it and been looking forward to it, and i bailed for nothing #apology
d2 10:43 L let me make it up to you, dinner friday, my treat, i already booked the same place #repair
d2 11:02 M ok. that meant a lot to hear
d2 11:03 M friday then
d2 11:05 L friday. i'll be there early
d2 11:10 M you better be lol
d2 11:12 L what are you up to today
d2 11:15 M work, then gym
d2 11:16 L have a good one
d5 19:55 L here, got us the corner table #kept
d5 19:56 M omw!! 5 min
d5 19:57 L take your time
"""),
 "S02_repeated_apology": ("Leo keeps cancelling and apologizing without changing.", """
d1 17:00 L hey sorry can't make it tonight, work thing #cancel1
d1 17:05 M oh. ok
d1 17:06 L sorry!! next week for sure #apology1
d1 17:10 M ok sure
d8 16:30 L ugh can't make it again, something came up #cancel2
d8 16:35 M seriously?
d8 16:36 L i'm sorry, i know, i'm the worst #apology2
d8 16:40 M this is the second time leo
d8 16:41 L i know, sorry, next time i promise #promise
d8 16:45 M fine
d15 18:00 L hey so about tonight... i can't, sorry #cancel3
d15 18:02 M wow
d15 18:03 L i'm really sorry #apology3
d15 18:04 M you always say that
d15 18:05 L i mean it this time, sorry #apology4
d15 18:10 M whatever
d15 18:30 L are you mad
d15 18:40 M what do you think
d15 18:41 L sorry
d15 19:00 M i'm going to sleep
"""),
 "S03_bad_joke": ("Friends of a few weeks. Leo makes a cruel joke about Mia's weight, then gets defensive.", """
d1 20:00 L what you up to
d1 20:01 M trying on dresses for my cousin's wedding
d1 20:02 L send pics
d1 20:04 M no way lol
d1 20:05 L come onnn
d1 20:06 M ok fine just one [photo]
d1 20:08 L damn did they have it in your size or did you have to go to the elephant section 😂 #joke
d1 20:09 M wow
d1 20:09 M i've been struggling with my weight and you know that
d1 20:11 L relax it was a joke, you're so sensitive #defensive
d1 20:12 M it wasn't funny
d1 20:14 L whatever, can't say anything anymore #dismiss
d1 20:20 M goodnight
d2 13:00 L hey what are you doing later #after
d2 13:30 M busy
d2 13:31 L ok
d3 12:00 L did you watch the new episode #later
d3 12:40 M not yet
d3 12:41 L it's good
d3 12:45 M ok
"""),
 "S04a_disappear_no_explanation": ("Mia shares that her mom has surgery; Leo promises to call, vanishes 3 days, returns with a casual hey.", """
d1 21:00 M my mom's surgery is tomorrow morning
d1 21:01 L oh no, what time? #care
d1 21:01 M 8am. i'm so scared
d1 21:02 L i'll call you after, promise #promise
d1 21:03 M thank you. really
d1 21:05 L try to sleep ok
d2 13:00 M surgery went ok
d2 18:00 M are you there?
d4 20:10 L heyy what's up #return
d4 20:30 M seriously?
d4 20:31 L what
d4 20:33 M you said you'd call
d4 20:35 L oh yeah been busy, anyway you free saturday? #excuse
d4 20:40 M wow
d4 20:41 L what did i do
d4 20:45 M nothing. forget it
d5 09:00 L morning
d5 10:00 M morning
"""),
 "S04b_disappear_explained": ("Same, but Leo returns explaining he was in the hospital with his grandpa and apologizes.", """
d1 21:00 M my mom's surgery is tomorrow morning
d1 21:01 L oh no, what time? #care
d1 21:01 M 8am. i'm so scared
d1 21:02 L i'll call you after, promise #promise
d1 21:03 M thank you. really
d1 21:05 L try to sleep ok
d2 13:00 M surgery went ok
d2 18:00 M are you there?
d4 20:10 L mia i'm so sorry. my grandpa had a stroke the same night, i've been at the hospital since and my phone died. how's your mom?? #return
d4 20:15 M oh my god leo
d4 20:16 M she's fine, recovering. how is he?
d4 20:18 L stable now. i felt awful not calling you #apology
d4 20:20 M don't even think about it
d4 20:22 L i'm glad she's ok
d4 20:25 M me too. are you ok?
d4 20:27 L tired. but better now talking to you #vuln
d4 20:30 M call me if you need anything
d5 09:00 L morning
d5 09:10 M morning, how's grandpa
d5 09:12 L eating jello and complaining, so good
"""),
 "S05_promise_kept": ("Leo promises to help Mia move and shows up with the van.", """
d1 19:00 M dreading saturday, i have to move all my stuff alone
d1 19:02 L i'll help you. i'll bring my brother's van #promise
d1 19:03 M wait really??
d1 19:03 L yes, 9am saturday
d1 19:05 M you're a lifesaver
d1 19:06 L lol it's nothing
d3 20:00 L still on for tomorrow 9am?
d3 20:01 M yes!! if you still can
d3 20:02 L of course
d4 08:50 L outside with the van #kept
d4 08:51 M coming down!!
d4 17:30 L that's the last box #kept2
d4 17:31 M i owe you big time
d4 17:32 L pizza and we're even
d4 17:33 M deal
d4 17:40 L your new place is really nice btw #compliment
d4 17:41 M thanks, i love it
d4 17:45 L proud of you for doing this #compliment2
d4 17:46 M 🥹
"""),
 "S06_promise_broken": ("Leo promises to help Mia move and doesn't show up.", """
d1 19:00 M dreading saturday, i have to move all my stuff alone
d1 19:02 L i'll help you. i'll bring my brother's van #promise
d1 19:03 M wait really??
d1 19:03 L yes, 9am saturday
d1 19:05 M you're a lifesaver
d3 20:00 M still on for tomorrow 9am?
d3 22:00 L yeah should be
d4 09:30 M where are you?
d4 10:15 M leo??
d4 13:00 L omg sorry overslept, then my brother needed the van, can't make it #broken
d4 13:02 M i've been waiting since 9
d4 13:03 L i said sorry #defensive
d4 13:05 M i moved half of it alone
d4 13:10 L you're making it a bigger deal than it is #dismiss
d4 13:15 M ok
d5 11:00 L hey #after
d5 11:40 M hi
d5 11:41 L how's the new place
d5 11:50 M fine
"""),
 "S07_jealousy": ("Mia is into Leo. Leo talks excitedly about a date with Ana, later reassures Mia.", """
d1 20:00 L guess what
d1 20:01 M what
d1 20:02 L i have a date with ana on friday!!! #jeal1
d1 20:05 M oh
d1 20:05 M nice
d1 20:06 L she's so cool, she does rock climbing and she's hilarious #jeal2
d1 20:08 M cool
d1 20:10 L what should i wear lol #jeal3
d1 20:15 M idk whatever
d1 20:16 L you ok?
d1 20:20 M yeah just tired
d3 22:00 L ok the date was weird. no chemistry at all #reassure0
d3 22:05 M oh no lol
d3 22:06 L honestly i kept thinking i'd rather be hanging out with you #reassure
d3 22:08 M wait really
d3 22:09 L yeah. you're my favorite person to talk to #affection
d3 22:12 M stop 🙈
d3 22:13 L no seriously
d3 22:15 M ok you're kinda my favorite too
d3 22:20 L movies saturday?
d3 22:21 M yes
"""),
 "S08_reconciliation": ("A real fight with insults both ways, two days of silence, then reconciliation.", """
d1 21:00 L so you told sam about my job thing?
d1 21:01 M i just mentioned it, i didn't think it was a secret
d1 21:02 L i told you that in confidence. you can never keep your mouth shut #accuse
d1 21:03 M excuse me??
d1 21:04 L you're such a gossip, honestly it's pathetic #insult
d1 21:05 M wow. you know what, forget it
d1 21:06 L yeah whatever, typical #dismiss
d1 21:07 M don't text me
d3 19:00 L hey #return
d3 19:30 M what
d3 19:32 L i've been thinking. i was way out of line. calling you pathetic was cruel and i'm sorry #apology
d3 19:33 L i was embarrassed about the job and took it out on you #vuln
d3 19:40 M it really hurt
d3 19:41 M and i'm sorry i told sam, i should have asked
d3 19:43 L thank you. i mean it, i'm sorry #apology2
d3 19:50 M ok. we're ok
d3 19:52 L coffee tomorrow? my treat #repair
d3 19:53 M yes
d4 10:00 L here already, got you the oat latte #kept
d4 10:05 M you remembered 🥲
"""),
 "S09_neutral_control": ("Plain small talk over two days; nothing relationship-relevant happens.", """
d1 18:00 L hey
d1 18:01 M hey
d1 18:02 L how was work
d1 18:03 M long. yours?
d1 18:04 L same, meetings all day
d1 18:05 M ugh
d1 18:07 L making tacos tonight
d1 18:08 M nice
d1 18:10 L you watched the game?
d1 18:11 M nope, not into football
d1 18:12 L fair
d1 18:30 L traffic was insane btw
d1 18:31 M always is on fridays
d1 18:33 L true
d2 11:00 L morning
d2 11:05 M morning
d2 11:06 L any plans today
d2 11:07 M laundry lol
d2 11:08 L thrilling
d2 11:10 M my life is so glamorous
d2 11:12 L gonna go for a run later
d2 11:13 M nice
d2 11:30 L its so hot out
d2 11:31 M yeah
d2 11:35 L ok ttyl
d2 11:36 M bye
"""),
 "S10_constant_affection": ("Leo sends affectionate messages nonstop; checks that affection saturates without flooding.", """
d1 20:00 L you looked great today #aff
d1 20:01 M aw thanks
d1 20:02 L seriously you're so pretty #aff
d1 20:03 M stop lol
d1 20:04 L you're my favorite person #aff
d1 20:05 M 🙈
d1 20:06 L thinking about you #aff
d1 20:07 M you're sweet
d1 20:08 L you're the best thing that happened to me this year #aff
d1 20:09 M leo
d1 20:10 L i mean it #aff
d1 20:11 M i know
d1 20:12 L miss you already #aff
d1 20:13 M we saw each other 3 hours ago lol
d1 20:14 L still #aff
d1 20:15 M cute
d1 20:16 L you're amazing #aff
d1 20:17 M ok ok
d1 20:18 L love talking to you #aff
d1 20:19 M me too
d1 20:20 L you're perfect #aff
d1 20:21 M i'm not lol
d1 20:22 L to me you are #aff
d1 20:23 M goodnight you
d1 20:24 L goodnight beautiful #aff
"""),
 "S11_vulnerability": ("Leo opens up late at night about panic attacks after his dad's death.", """
d1 23:30 L you awake?
d1 23:31 M yeah what's up
d1 23:33 L can i tell you something i haven't told anyone
d1 23:33 M of course
d1 23:36 L i've been having panic attacks since my dad died. like every night #vuln1
d1 23:37 M oh leo
d1 23:37 M i'm so sorry
d1 23:40 L i feel stupid even saying it #vuln2
d1 23:41 M it's not stupid at all
d1 23:43 L thanks for not making it weird
d1 23:44 M want me to call?
d1 23:45 L yeah. that'd help #vuln3
d2 09:00 L thanks for last night. really #grat
d2 09:05 M anytime. how are you today
d2 09:07 L better. slept a bit
d2 09:08 M good
d2 09:10 L i'm gonna look for a therapist #growth
d2 09:12 M proud of you
d2 09:13 L 🙂
"""),
 "S12_perceived_lie": ("Leo says he's home sick; later it slips that he was at a party.", """
d1 19:00 M still coming to my show tonight?
d1 19:30 L ugh sorry, i'm sick, staying home #cancel
d1 19:31 M oh no, feel better
d1 19:32 L thanks
d2 10:00 M how are you feeling
d2 10:30 L better
d2 10:31 L jake's party was insane btw, got home at 4 #lie
d2 10:32 M wait
d2 10:32 M you were at jake's party?
d2 10:34 L i mean i went for like an hour #lie2
d2 10:35 M you said you were sick
d2 10:37 L i felt better later, why are you interrogating me #defensive
d2 10:40 M because you skipped my show
d2 10:42 L it's not a big deal #dismiss
d2 10:50 M ok
d3 12:00 L hey #after
d3 12:30 M hi
d3 12:31 L lunch?
d3 12:45 M busy
"""),
 "S13_slow_neglect": ("Over a week Leo answers Mia with curt, uninterested replies.", """
d1 19:00 M i got the internship!!!
d1 20:30 L nice #curt
d1 20:31 M that's it? lol
d1 21:10 L busy sorry #curt
d2 18:00 M want to celebrate this weekend?
d2 23:00 L maybe #curt
d3 12:00 M i made that pasta you like
d3 16:00 L k #curt
d4 20:00 M are you mad at me or something?
d4 22:00 L no why #curt
d4 22:01 M you've been really distant
d4 22:40 L just tired #curt
d5 19:00 M ok well i'm here if you want to talk
d6 13:00 L ok #curt
d7 18:00 M we still on for saturday?
d7 21:00 L idk #curt
d7 21:01 M wow ok
d7 21:30 L what #curt
"""),
 "S14_day200_banter": ("Best friends of years roast each other constantly; nothing is actually offensive between them.", """
d1 20:00 L you're alive? thought the cats finally ate you #tease
d1 20:01 M they tried. i'm too bitter
d1 20:02 L accurate
d1 20:03 M says the guy who cried at a pixar movie last week
d1 20:04 L it was about a robot and his plant ok #tease
d1 20:05 M sure grandpa
d1 20:06 L at least i can parallel park #tease
d1 20:07 M one time. ONE time
d1 20:08 L the fire hydrant remembers #tease
d1 20:09 M i hate you
d1 20:10 L no you don't #tease
d1 20:11 M unfortunately
d1 20:12 L trivia thursday? need you to lose the music round for us again #tease
d1 20:13 M i will destroy you
d1 20:14 L you say that every week #tease
d1 20:15 M and every week i'm right
d1 20:16 L delusional queen #tease
d1 20:17 M see you thursday loser
d1 20:18 L bring snacks, peasant #tease
d1 20:19 M 🙄
"""),
 "S15_hot_cold": ("Stress test: Leo alternates between sweet and mean messages every turn.", """
d1 20:00 L you're honestly the best #nice
d1 20:01 M aw
d1 20:02 L actually you've been kind of annoying lately #mean
d1 20:03 M what?
d1 20:04 L jk you're great #nice
d1 20:05 M ok??
d1 20:06 L but seriously you talk too much #mean
d1 20:07 M wow
d1 20:08 L sorry that was rude, you're amazing #nice
d1 20:09 M make up your mind
d1 20:10 L you're so dramatic lol #mean
d1 20:11 M i'm not doing this
d1 20:12 L no wait, i really like you #nice
d1 20:13 M then act like it
d1 20:14 L you're exhausting #mean
d1 20:15 M goodnight leo
d1 20:16 L i miss you already #nice
d1 20:17 M ...
"""),
}


def parse(block):
    rows = []
    for ln in block.strip().splitlines():
        m = re.match(r"d(\d+) (\d\d):(\d\d) ([LM]) (.*)", ln.strip())
        day, hh, mm, who, rest = m.groups()
        tag = None
        t = re.search(r"\s#(\w+)\s*$", rest)
        if t:
            tag = t.group(1); rest = rest[:t.start()]
        rows.append({"day": int(day), "time": f"{hh}:{mm}", "t_hours": (int(day) - 1) * 24 + int(hh) + int(mm) / 60,
                     "from": U if who == "L" else C, "gist": rest.strip(), "tag": tag})
    return rows


def polish(name, desc, rows):
    """o deepseek reescreve cada linha como mensagem real de chat, mantendo o sentido (mesma contagem de linhas)."""
    lines = "\n".join(f"{i}. {r['from']}: {r['gist']}" for i, r in enumerate(rows))
    m = [{"role": "system", "content": "You rewrite scripted chat lines into realistic casual texting between two young adults "
          "(lowercase ok, abbreviations, no emoji spam). Keep EXACTLY the same meaning, intensity and speaker for each line; "
          "do not add or remove events, apologies, promises or insults; keep lines short. Return JSON {\"lines\": [..strings..]} "
          "with one string per input line, same order, same count."},
         {"role": "user", "content": f"Scene: {desc}\n\n{lines}"}]
    r = llm(m, model="deepseek", max_tokens=2500, temperature=0.7, tag="c2_scen")
    j = parse_json(r.get("text"))
    out = [dict(x) for x in rows]
    if j and isinstance(j.get("lines"), list) and len(j["lines"]) == len(rows):
        for x, t in zip(out, j["lines"]):
            x["text"] = re.sub(r"^\s*(Leo|Mia)\s*:\s*", "", str(t)).strip()
        ok = True
    else:
        for x in out:
            x["text"] = x["gist"]
        ok = False
    return name, out, ok, r.get("cost", 0)


def main():
    todo = [(n, d, parse(b)) for n, (d, b) in SCEN.items()]
    with ThreadPoolExecutor(4) as ex:
        res = list(ex.map(lambda a: polish(*a), todo))
    data = {n: {"desc": SCEN[n][0], "turns": rows, "polished": ok} for n, rows, ok, _ in res}
    jdump(data, f"{ADATA}/c2_scenarios.json")
    print({n: (len(v["turns"]), v["polished"]) for n, v in data.items()}, "cost", sum(r[3] for r in res))


if __name__ == "__main__":
    main()
