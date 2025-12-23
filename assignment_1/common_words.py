##- the following top fake and real words are *derived* from top_20_fake & top_20_true files

# Fake top words: derived from top_20_fake file
fake = [
    "trump",
    "said",
    "just",
    "people",
    "president",
    "video",
    "like",
    "image",
    "donald",
    "time",
    "featured",
    "new",
    "news",
    "right",
    "make",
    "know",
    "obama",
    "america",
    "don",
    "going",
    "way",
    "did",
    "american",
    "watch",
    "say",
    "white",
    "told",
    "year",
    "state" "house",
]
# True top words: derived from top_20_true file
real = [
    "reuters",
    "said",
    "president",
    "trump",
    "donald",
    "told",
    "state",
    "government",
    "year",
    "new",
    "people",
    "states",
    "united",
    "house",
    "republican",
    "including",
    "week",
    "country",
    "tuesday",
    "election",
    "wednesday",
    "time",
    "did",
    "years",
    "thursday",
    "statement",
    "national",
    "security",
    "minister",
    "called",
]

common_items_set = set(fake) & set(real)


def get_pure_top_real_words_set():
    return set(real) - common_items_set


def get_pure_top_fake_words_set():
    return set(fake) - common_items_set
