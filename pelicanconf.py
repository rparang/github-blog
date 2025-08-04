AUTHOR = 'Reza Parang'
SITENAME = 'Reza Parang'
SITEURL = ""
RELATIVE_URLS = True

PATH = "content"

TIMEZONE = 'America/Los_Angeles'

DEFAULT_LANG = 'en'

THEME = 'themes/custom' # Must include to make with your custom files

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = (
    ("Pelican", "https://getpelican.com/"),
)

# Social widget
SOCIAL = (
    ("You can add links in your config file", "#"),
)

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
# RELATIVE_URLS = True

# Disable category pages
CATEGORY_SAVE_AS = ''
CATEGORY_URL = ''

# Disable author pages
AUTHOR_SAVE_AS = ''
AUTHOR_URL = ''

# Disable tag pages
TAG_SAVE_AS = ''
TAG_URL = ''

# Disable archives and indexes
ARCHIVES_SAVE_AS = ''
YEAR_ARCHIVE_SAVE_AS = ''
MONTH_ARCHIVE_SAVE_AS = ''
DAY_ARCHIVE_SAVE_AS = ''


SUMMARY_MAX_LENGTH = 150  # or whatever number of words you want

# Prevent Pelican from generating individual pages like /categories.html, /authors.html, /tags.html
DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_PAGES_ON_MENU = False
AUTHOR_SAVE_AS = ''
CATEGORY_SAVE_AS = ''
TAG_SAVE_AS = ''
TAGS_SAVE_AS = ''
CATEGORIES_SAVE_AS = ''
AUTHORS_SAVE_AS = ''
ARCHIVES_SAVE_AS = ''

STATIC_PATHS = ['images'] # Anything in STATIC_PATHS gets copied as-is to output/ folder