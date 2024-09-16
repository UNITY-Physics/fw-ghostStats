import argparse
import sys

from .cmd import download_all_ref_data, update_sidecar, warp_rois


def main():
    GHOST_parser()

class GHOST_parser(object):

    def __init__(self):
        method_list = [method for method in dir(self) if method.startswith('entrypoint')]
        
        method_str=''
        for method in method_list:
            dstr = eval(f'self.{method}.__doc__')
            nice_name = method.replace('entrypoint_', '')

            method_str += "\t{:25s}{:25s}\n".format(nice_name, dstr)

        parser = argparse.ArgumentParser(description='GHOST: A framework for phantom analysis in the UNITY project',
                                         usage=f'''ghost <command> [<args>]

    Available commands are
{method_str}

    ''')
        
        parser.add_argument('command', help='Subcommand to run')
        args = parser.parse_args(sys.argv[1:2])

        # Check if the object (the class) has a function with the given command name
        if not hasattr(self, "entrypoint_" + args.command):
            print('Unrecognized command')
            parser.print_help()
            exit(1)

        # Call the method
        getattr(self, "entrypoint_" +args.command)()

    def entrypoint_warp_rois(self):
        """Warp ROIS"""
        parser = argparse.ArgumentParser(description='Warp ROIs to target image',
                                         usage='ghost warp_rois <input> [<args>]]')
        
        parser.add_argument('input', type=str, help='Input image')
        parser.add_argument('-s', '--seg', required=True, type=str, help='Segmentation')
        parser.add_argument('-w', '--weighting', type=str, default='T1', help='Phantom weighting')
        parser.add_argument('-o', '--out', type=str, default=None, help='Output basename (default is input basename)')
        parser.add_argument('-p', '--phantom', type=str, default='Caliber137', help='Phantom model to use')
        parser.add_argument('--syn', action='store_true', help='Use deformable SyN registration to template')
        parser.add_argument('--vol', type=int, default=None, help='Volume to use (default is last volume)')
        parser.add_argument('--save_xfm', action='store_true', help='Save transform for later use. Specify names with --xfm_out. Default is input basename')
        parser.add_argument('--xfm_aff_in', type=str, default=None, help='Input file name for affine template->image transform')
        parser.add_argument('--xfm_syn_in', type=str, default=None, help='Input file name for syn template->image transform')
        parser.add_argument('--xfm_out', type=str, default=None, help='Filename to save calculated transform(s)')
        
        args = parser.parse_args(sys.argv[2:])
        warp_rois(input=args.input, output=args.out, seg=args.seg, 
                  weighting=args.weighting, vol=args.vol, phantom_model=args.phantom, 
                  do_syn=args.syn, xfm_out_name=args.xfm_out, xfm_aff_in=args.xfm_aff_in, xfm_syn_in=args.xfm_syn_in, save_xfm=args.save_xfm)


    def entrypoint_setup(self):
        """Download data"""
        parser = argparse.ArgumentParser(description='Setup repo and download data',
                                         usage='ghost setup')
        args = parser.parse_args(sys.argv[2:])
        download_all_ref_data(args)


    def update_sidecar(self):
        """Update json sidecar info from dicom tags"""
        parser = argparse.ArgumentParser(description='Update sidecar json files with info from dicom files')
        parser.add_argument('dicom_dir', help='Directory containing dicom files')
        parser.add_argument('json_dir', help='Directory containing json files')
        parser.add_argument('-m', '--matches', help='File containing matches between dicom tags and json fields', required=True)
        parser.add_argument('-v', '--verbose', help='Verbose output', action='store_true')
        
        args = parser.parse_args(sys.argv[2:])
        update_sidecar(args)


if __name__ == '__main__':
    main()
