# Create Setup.py file to pull in Matplotlib and Python-tk for Dataflow

from distutils.command.build import build as _build
import subprocess
import setuptools

class build(_build):
    sub_commands = _build.sub_commands + [('CustomCommands', None)]

CUSTOM_COMMANDS = [
                   ['apt-get', 'update'],
                   ['apt-get', '--assume-yes', 'install', 'python-tk'],
                   ['pip', 'install', 'matplotlib==2.0.2']
]

class CustomCommands(setuptools.Command):
    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def RunCustomCommand(self, command_list):
        print 'Running command: %s' % command_list
        p = subprocess.Popen(
                             command_list,
                             stdin = subprocess.PIPE,
                             stdout = subprocess.PIPE,
                             stderr = subprocess.STDOUT
                            )
        stdout_data, _ = p.communicate()
        print 'Command output: %s' % stdout_data
        if p.returncode != 0:
            raise RuntimeError('Command %s failed: exit code: %s' % (command_list, p.returncode))

    def run(self):
        for command in CUSTOM_COMMANDS:
            self.RunCustomCommand(command)

setuptools.setup(
    name = 'mikebranch',
    version = '1.0',
    description = 'Industry-Image-Generator',
    packages = setuptools.find_packages(),
    cmdclass = {
                'build': build,
                'CustomCommands': CustomCommands,
               }
)