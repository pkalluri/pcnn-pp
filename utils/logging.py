"""
Miscellaneous helper functions
"""

from inspect import currentframe, getframeinfo

def log(s):
	frame = currentframe()
	filename = getframeinfo(frame).filename
	line_number = frame.f_back.f_lineno
	print('{fn} {no}: {s}'.format(fn=filename, no=line_number, s=s))



# Test log function
if __name__ == '__main__':
	log('testing')