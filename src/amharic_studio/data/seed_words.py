"""Seed wordlist so the corrector is useful before any corpus is loaded.

This is deliberately small and high-confidence: function words, pronouns, numerals,
everyday vocabulary, and the religious and historical register that dominates older
Ethiopian printing. It is *not* a substitute for a real corpus — Amharic is heavily
affixed, so most inflected forms will be missing until a corpus is ingested via
``tools/build_lexicon.py``. Both spellings are listed where a word is commonly written
two ways (ሰላሳ / ሠላሳ), since the point is recognition, not prescription.

Frequencies are rough magnitude tiers used only to break ties between candidates.
"""

from __future__ import annotations

_FUNCTION = 10000
_VERY_COMMON = 3000
_COMMON = 800
_MODERATE = 200

_TIERS: dict[int, str] = {
    _FUNCTION: """
    እና ወይም ግን ነገር ስለ ስለዚህ ሆኖም ደግሞ ብቻ ሁሉ ሌላ ሌሎች እንደ ወደ ላይ ውስጥ ጋር
    በኋላ በፊት ድረስ ሲሆን ማለት ማለትም ይህ ያ እነዚህ እነዚያ ይኸ ያም ይኽ
    ነው ናት ናቸው ነኝ ነህ ነሽ ነን አይደለም አለ አለች አሉ ነበር ነበረ ነበሩ ነበረች
    እኔ አንተ አንቺ እሱ እሷ እኛ እናንተ እነሱ እርሱ እርሷ እርስዎ ራሱ ራሷ ራሳቸው
    ማን ምን የት መቼ እንዴት ለምን ስንት የቱ
    አዎ አይ አልነበረም የለም አለው አላት አላቸው ኧረ
    """,
    _VERY_COMMON: """
    ሰው ሰዎች ልጅ ልጆች አገር ሀገር ሃገር ኀገር ከተማ ቤት ቤቶች ቦታ ዓለም አለም
    ቀን ሌሊት ማታ ጠዋት ሰዓት ደቂቃ ዓመት አመት ወር ሳምንት ጊዜ ዘመን ዛሬ ትናንት ነገ አሁን
    ውሃ ውኃ እሳት አየር መሬት ሰማይ ፀሐይ ጸሐይ ጨረቃ ኮከብ ብርሃን ጨለማ
    ታሪክ መጽሐፍ መጽሔት ወረቀት ትምህርት ተማሪ መምህር ሥራ ስራ ሠራተኛ ገንዘብ ብር
    ምግብ እንጀራ ወጥ ቡና ሻይ ወተት ስጋ ዳቦ ጨው ስኳር በርበሬ
    አባት እናት ወንድም እህት ሚስት ባል ቤተሰብ ዘመድ ጓደኛ ወዳጅ
    ትልቅ ትንሽ ረጅም አጭር ጥሩ መጥፎ አዲስ አሮጌ ብዙ ጥቂት ከባድ ቀላል ውብ ቆንጆ
    ጥቁር ነጭ ቀይ አረንጓዴ ሰማያዊ ቢጫ
    አንድ ሁለት ሶስት ሦስት አራት አምስት ስድስት ሰባት ስምንት ዘጠኝ አስር አስራ
    ሃያ ሀያ ሰላሳ ሠላሳ አርባ አምሳ ሃምሳ ሐምሳ ስልሳ ስድሳ ሰባ ሰማንያ ዘጠና መቶ ሺህ ሺ ሚሊዮን
    መጀመሪያ መጨረሻ ክፍል ምዕራፍ ገጽ ቁጥር ስም ቃል ቋንቋ አማርኛ ግዕዝ
    """,
    _COMMON: """
    እግዚአብሔር አምላክ ጌታ ኢየሱስ ክርስቶስ መንፈስ ቅዱስ ቅድስት ማርያም መልአክ
    ገነት ሲኦል ጸሎት ጾም ፆም በዓል በአል ካህን ወንጌል ኦሪት መዝሙር ሃይማኖት ሐይማኖት እምነት
    ኃጢአት ኀጢአት ሐጢአት ምሕረት ምህረት ቡራኬ አሜን ጳጳስ ደብተራ ገዳም ቤተክርስቲያን መስቀል
    ንጉሥ ንጉስ ንግሥት ንግስት መንግሥት መንግስት አጼ አፄ ራስ ደጃዝማች ፊታውራሪ ጦር ወታደር
    ጦርነት ሰላም ሠላም ፍቅር ጥላቻ ደስታ ሐዘን ሀዘን ተስፋ ፍርሃት ችግር መፍትሔ መፍትሄ
    ራስ ዓይን አይን ጆሮ አፍ አፍንጫ እጅ እግር ልብ ደም ጭንቅላት ጸጉር ፀጉር ጥርስ
    መሄድ መምጣት መብላት መጠጣት መተኛት መነሳት ማየት መስማት መናገር መጻፍ ማንበብ
    መስራት መሥራት መማር ማስተማር መውደድ መጥላት መስጠት መውሰድ መግዛት መሸጥ
    ማወቅ መርሳት መጀመር መጨረስ መሆን ማድረግ መያዝ መልቀቅ መክፈት መዝጋት
    ሄደ መጣ በላ ጠጣ አየ ሰማ ተናገረ ጻፈ አነበበ ሰራ ሠራ ተማረ አስተማረ ወደደ
    ሰጠ ወሰደ ገዛ ሸጠ አወቀ ጀመረ ጨረሰ ሆነ አደረገ ያዘ ከፈተ ዘጋ ሞተ ተወለደ
    ይላል ትላለች ይሆናል ይችላል ትችላለች እንችላለን አልችልም
    ኢትዮጵያ አዲስ አበባ አማራ ኦሮሚያ ትግራይ ጎንደር ጎጃም ወሎ ሸዋ ሐረር ኤርትራ
    አፍሪካ አውሮፓ አሜሪካ እስራኤል ግብጽ ግብፅ ሱዳን
    ትግርኛ ኦሮምኛ እንግሊዝኛ ዓረብኛ አረብኛ
    """,
    _MODERATE: """
    ሁልጊዜ አንዳንድ ፈጽሞ በጣም ቶሎ ቀስ ደግሞም እንዲሁ ወዲያው ቀደም ኋላ ፊት
    ጉዳይ ሁኔታ ምክንያት ውጤት ዓላማ አላማ ሀሳብ ሐሳብ ኃሳብ እውነት ውሸት ጥያቄ መልስ
    ሕዝብ ህዝብ ማህበረሰብ ማኅበረሰብ ባህል ባሕል ልማድ ሥርዓት ስርዓት ሕግ ህግ ፍርድ ዳኛ
    ጤና ሕክምና ህክምና ሐኪም ሀኪም በሽታ መድኃኒት መድሃኒት ሆስፒታል
    ገበያ ሱቅ ንግድ ነጋዴ ገበሬ እርሻ እህል ዘር ማሳ በሬ ላም ፍየል በግ ዶሮ ፈረስ አህያ
    ዛፍ አበባ ቅጠል ሥር ስር ደን ተራራ ወንዝ ባህር ባሕር ሐይቅ ሀይቅ ደሴት በረሃ
    መኪና አውሮፕላን ባቡር መንገድ ድልድይ በር መስኮት ክፍል ወንበር ጠረጴዛ አልጋ
    ልብስ ጫማ ቀሚስ ሱሪ ኮፍያ ሸማ ጋቢ ነጠላ
    ሙዚቃ ዘፈን ጭፈራ በገና ከበሮ ማሲንቆ ዋሽንት ክራር
    ግጥም ቅኔ ተረት ምሳሌ አባባል ወግ ልቦለድ ድርሰት ደራሲ ገጣሚ አዘጋጅ አሳታሚ
    ጥናት ምርምር ሳይንስ ሒሳብ ሂሳብ ፊደል ሆሄ አኃዝ አሃዝ ነጥብ
    """,
}


def _parse() -> list[tuple[str, int]]:
    seen: dict[str, int] = {}
    for freq, blob in _TIERS.items():
        for word in blob.split():
            # Keep the highest tier if a word appears in more than one block.
            if seen.get(word, 0) < freq:
                seen[word] = freq
    return sorted(seen.items())


SEED_WORDS: list[tuple[str, int]] = _parse()

__all__ = ["SEED_WORDS"]
