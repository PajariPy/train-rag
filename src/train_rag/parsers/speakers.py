"""Slack speaker lines, and the first names of the people who post in the workspace.

Each Slack message starts with a speaker line in one of several forms (`maria: ...`,
`Name (Role): ...`, `[alex] 1789456789: ...`); these patterns and helpers split a thread
into messages.

The names come from the thread starters (the first speaker of each thread, almost always
a person) across all Slack threads: the first part of handles like `maria_cs` or `alex.k`,
used by at least 20 threads. Bots, role words (`support`, `ops`, `incident`) and names
that are also config keys (`max_tokens:`, `grace_period:`) are left out. They tell a
speaker line written another way (` dave_oncall:`, `maria gonzalez:`) from a `key:` line
inside a code block.
"""

import re
from collections.abc import Callable

from train_rag.models.parsed import SlackMessage

SPEAKER_LINE = re.compile(
    r"^(?P<speaker>[A-Za-z][\w.\-]*(?: [A-Z][\w.\-]*){0,2})"
    r"(?: \((?P<role>[^)\n]{1,40})\))?:\s?(?P<text>.*)$"
)
LOOSE_SPEAKER_LINE = re.compile(
    r"^\s*(?P<speaker>[A-Za-z][\w.\-]*(?: [A-Za-z][\w.\-]*){0,2})\s*"
    r"(?:\((?P<role>[^)\n]{1,40})\)|\s-\s(?P<dash_role>[A-Za-z][\w &/-]{0,30}))?"
    r":\s?(?P<text>.*)$"
)
BRACKETED_SPEAKER_LINE = re.compile(
    r"^\s*\[(?P<speaker>[A-Za-z][\w.\- ]{0,40})\]\s*(?:\d{9,10}(?:\.\d+)?)?\s*:\s?(?P<text>.*)$"
)
BRACED_SPEAKER_LINE = re.compile(r"^\s*\{(?P<speaker>[A-Za-z][\w.\- ]{0,40})\}\s*:\s?(?P<text>.*)$")
HANDLE_SEPARATOR = re.compile(r"[_.\-]")
KNOWN_FIRST_NAMES = frozenset(
    """
    aaron ada adele aditi aditya adrian aisha aj ajay akash akshay alan alana alec alejandra
    alejandro alex alexa alexander ali alia alice alicia alina aline alisa alisha allison alyssa
    amal amanda amelia amina amir amira amy ana andre andrea andrew ani anika anil anita anjali
    anna anne annie anu anya ara ari aria arjun arun arya ash asha ashley ashok ashwin ava avery
    aya bea becca ben benji beth bob bryan cam camila camille cara carl carla carlos carmen
    carol caroline casey cass cassie charlie chen cheng chloe chris ciara claire clara clare
    claudia cleo cora cory courtney dalia dan dana dani daniel danielle dara daria dario dave
    david dena derek devin devon diane diego dina dinesh dmitri drew dylan eden ela elaine
    eleanor elen elena eleni eli ella elle ellen ellie elliot eloise elsa elyse emi emily emma
    eric erica erin ethan eva evan eve evelyn faye felix fiona fran frank gabe gina hana
    hannah hao harper harry hector helen hema henry hiro ian ibrahim ida indra ingrid ira irene
    irina iris isabel isha ishaan ivan ivy izzy jade jae jake jamal james jamie jana jane
    janelle janet jared jas jasmin jason jasper javier jay jeff jen jena jenna jenny jess jesse
    jessica jia jin jo joan joanne joe joel john jon jonah jonas jonathan jordan jordana jordyn
    jorge josh juan jules julia julian julie jun june juno kai kara karen kari karim kat kate
    katie kayla keiko keisha kelly kelsey ken kenji kevin kim kira kiran kyle lana lara laura
    lauren leah lee leigh leila lena leo lex li lia liam liang lila lily lin lina linda linh
    lisa liz logan luca lucas lucy luis lydia maddie maddy mae maeve maggie mai maia mara marc
    marco marcos marcus margo mari maria mariam marie marin marina marisa marisol marissa marta
    mason mateo matt maya meera megan mei mel melissa mia mica micah michael michelle miguel
    mika mikaela mike mila miles mina mira miranda miriam mohit molly mona monica morgan nadia
    naomi natalie nate naveen neal neel neha neil nia nick nico nicole nikhil nina nisha noah
    noelle noor nora olga oliver olivia omar oscar owen pat paul paula peter pranav pri priya
    priyanka rachael rachel rae rafa rafael rafi rahul raj raja rajesh ravi rebecca reena rene
    renee rhea ria riley rina rita riya rohan rohit rory rosa rowan ruth ryan sachin sahil sam
    samantha sami samir samira samuel sana sandeep sanjay santiago sara sarah sasha sean serena
    shreya simon sofia sophia sophie sylvia sylvie talia tanvi tanya tara tariq tasha tay tayla
    taylor tess tessa theo tia tian tim tina tom tomas toni tony tori tyler vanessa vera vik
    vikram violet viv vivian will yara yasmin yuki zach zane zara zoe
    """.split()
)

LineMatcher = Callable[[str], re.Match[str] | None]


def known_speaker_line(line: str) -> re.Match[str] | None:
    """Match a strict speaker line, or a loose one from a known person or a bot.

    Code blocks hold `name:` and `enabled:` lines that look like loose speaker lines;
    the first name of the speaker (`dave` in `dave_oncall`) tells them apart. Names
    split by a stray space (`ev an`, `mar ia`) are joined before the lookup, and
    speakers ending in `bot` (`deploy-bot`) count too.

    Args:
        line: One thread line.

    Returns:
        The match, or None when the line does not start a message.
    """
    match = SPEAKER_LINE.match(line) or LOOSE_SPEAKER_LINE.match(line)
    if match is None:
        return None

    speaker = match.group("speaker").lower()
    first_names = {
        HANDLE_SEPARATOR.split(candidate)[0]
        for candidate in (speaker.split()[0], speaker.replace(" ", ""))
    }
    if match.re is SPEAKER_LINE or first_names & KNOWN_FIRST_NAMES or speaker.endswith("bot"):
        return match

    return None


def strict_speaker_messages(lines: list[str]) -> list[SlackMessage] | None:
    """Split lines into messages when at least one line is a strict speaker line.

    Args:
        lines: Thread lines.

    Returns:
        The messages in order, or None when no line strictly starts with a speaker, so
        that the looser formats are tried.
    """
    if not any(SPEAKER_LINE.match(line) for line in lines):
        return None

    return speaker_messages(lines, known_speaker_line) or None


def speaker_messages(lines: list[str], speaker_line: LineMatcher) -> list[SlackMessage]:
    """Split lines into messages; lines without a speaker continue the previous one.

    Args:
        lines: Thread lines.
        speaker_line: Match of a line that starts a message, with `speaker`, `text`
            and optional `role` or `dash_role` groups.

    Returns:
        The messages in order.
    """
    messages: list[SlackMessage] = []
    for line in lines:
        match = speaker_line(line)
        if match is not None:
            groups = match.groupdict()
            messages.append(
                SlackMessage(
                    speaker=match.group("speaker"),
                    role=groups.get("role") or groups.get("dash_role"),
                    text=match.group("text").strip(),
                )
            )
        elif messages and line.strip():
            messages[-1].text = f"{messages[-1].text}\n{line}"

    return messages
