# TIME PATTERNS
timePattern1 = '(\d{1,2}:\d{2}) - (\d{1,2}:\d{2})'  # dd:dd - dd:dd
am_pm = '(AM|PM|am|pm|A\.M\.|P\.M\.|a\.m\.|p\.m\.| AM| PM| am| pm| A\.M\.| P\.M\.| a\.m\.| p\.m\.)'
timePattern2 = "(\d{1,2}:\d{2}" + am_pm + ') - (\d{1,2}:\d{2}' + am_pm + ')'  # dd:dd am - dd:dd am
timePattern3 = '\d{1,2}:\d{2}'  # dd:dd || d:dd
timePattern4 = '(' + timePattern3 + am_pm + ')'  # dd:dd AM
timePattern5 = '(' + timePattern3 + ')' + ' - ' + timePattern4
timePattern6 = timePattern4 + ' - (' + timePattern3 + ')'

time_regx_str = r'\b([0-9]{1,2}(?::[0-9]{2}\s?(?:AM|PM|am|pm|a\.m|p\.m)|:[0-9]{2}|\s?(?:AM|PM|am|pm|a\.m|p\.m)))\b'
speaker_regx_str = r'(?:\b(?:Speaker|Name|Who|SPEAKER|NAME|WHO|SPREAKER)\b:\s*)(.*?,|.*)'
# speaker_regx_str = r'(?:\b(?:Speaker|Name|Who)\b:\s*)(.*?|.*)(?:,|-|\/)'
header_body_regx_str = r'([\s\S]+(?:\b.+\b:.+\n\n|\bAbstract\b:))([\s\S]*)'
knownLocationRegxStr = '<location>(.+)<\/location>'
knownSpeakersRegxStr = '<speaker>(?:Dr|Mr|Ms|Mrs|Prof|Sir|Professor)?\.?\s?([a-zA-Z ]+),?\s?(?:PhD)?<\/speaker>'
deadTag = '<\/sentence>'
deadTag1 = '<\/paragraph>'

location_regx_str = r'(?:\b(?:Place|Location|Where)\b:\s*)(.*)'
pos_location_regx_str = r'((?:(?:(?:\w*?{\*(?:NNP|CD)\*})|(?:room{\*.+?\*}))\s*)*)'
pos_tags_regx_str = r'{\*.+?\*}'

# paragraphRegex = r'(?s)((?:[^\n][\n]?)+)'
paragraphRegex = r'(?<=\n\n)(?:(?:\s*\b.+\b:(?:.|\s)+?)|(\s{0,4}[A-Za-z0-9](?:.|\n)+?\s*))(?=\n\n)'
not_sentence_regx_str = r'^[A-Za-z0-9](?:.|\n)+(?:\.|\?|!|:)$'

topic_regx_str = r'(?:\b(?:Topic)\b:\s*)(.*)'
special_char_regx_str = r'([^a-zA-Z ]+?)'