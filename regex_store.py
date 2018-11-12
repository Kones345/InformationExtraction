#TIME PATTERNS
timePattern1 = '(\d{1,2}:\d{2}) - (\d{1,2}:\d{2})' # dd:dd - dd:dd
am_pm = '(AM|PM|am|pm|A\.M\.|P\.M\.|a\.m\.|p\.m\.| AM| PM| am| pm| A\.M\.| P\.M\.| a\.m\.| p\.m\.)'
timePattern2 = "(\d{1,2}:\d{2}" + am_pm + ') - (\d{1,2}:\d{2}' + am_pm + ')'#dd:dd am - dd:dd am
timePattern3 =  '\d{1,2}:\d{2}' #dd:dd || d:dd
timePattern4 = '(' + timePattern3 + am_pm + ')'#dd:dd AM
timePattern5 = '(' + timePattern3 + ')' + ' - ' + timePattern4
timePattern6 = timePattern4 + ' - (' + timePattern3 + ')'


knownLocationRegxStr = '<location>(.+)<\/location>'
knownSpeakersRegxStr = '<speaker>(?:Dr|Mr|Ms|Mrs|Prof|Sir|Professor)?\.?\s?([a-zA-Z ]+),?\s?(?:PhD)?<\/speaker>'
deadTag = '<\/sentence>'
deadTag1 = '<\/paragraph>'