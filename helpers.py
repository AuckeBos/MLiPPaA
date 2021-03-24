def write_log(msg, error=False, kill=False):
    """
    Write a log to stdout
    :param msg: The log to write
    :param error: Is info or error? Defines the prefix
    :param kill: If is true, exit after log
    """
    prefix = 'E' if error else 'I'
    print(f'[{prefix}] - {msg}')
    if kill:
        exit()
