import subprocess as sp

def main():
    try:
        sp.call(['sphinx-build', '-b', 'html', 'doc', 'doc/_build'])
    except Exception as e:
        print('Could not build the documentation. Is sphinx installed?')

    print('Open doc/_build/index.html to view the help')
    
if __name__ == '__main__':
    main()