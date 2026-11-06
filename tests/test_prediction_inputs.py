from pathlib import Path
import sys,tempfile,unittest
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fire_r2.prediction import display_images


class PredictionInputTests(unittest.TestCase):
    def test_display_orientation_is_explicit_and_source_bytes_unchanged(self):
        from PIL import Image
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as tmp:
            p=Path(tmp)/'x.jpg';im=Image.new('RGB',(20,40),'red');exif=im.getexif();exif[274]=6;im.save(p,exif=exif)
            before=p.read_bytes();views=display_images([p])
            self.assertEqual(views[0].size,(40,20));self.assertEqual(views[0].mode,'RGB');self.assertEqual(p.read_bytes(),before)
