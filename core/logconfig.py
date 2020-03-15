import logging.config

_BASE_FORMAT = '%(asctime)-15s %(levelname)-5s %(module)-5s [%(funcName)s] %(message)-15s'

LOGGER_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters':
    {
        'simple_format':
        {
            'format': _BASE_FORMAT
        }
    },
    'handlers':
    {
        'default_console':
        {
            'class': 'logging.StreamHandler',
            'formatter': 'simple_format',
            'stream': 'ext://sys.stdout'
        },
        'default_file':
        {
            'class': 'logging.FileHandler',
            'formatter': 'simple_format',
            'filename': 'log.txt',
            'mode': 'w',
            'encoding': 'utf-8'
        }
    },
    'root':
    {
        'level': 'INFO',
        'handlers': ['default_console', 'default_file']
    }

}
