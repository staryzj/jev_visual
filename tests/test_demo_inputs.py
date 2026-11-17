import base64
import io
from pathlib import Path
import sys
import unittest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visualjev.demo.server import validate


class Inputs(unittest.TestCase):
    def payload(self):
        stream = io.BytesIO()
        Image.new('RGB', (24, 16), '#e3ad78').save(stream, 'PNG')
        return dict(image='data:image/png;base64,' + base64.b64encode(stream.getvalue()).decode(),
                    question=' Where is the object? ', candidates=['left', 'right'])

    def test_uploaded_pixels_and_order_are_preserved(self):
        image, question, choices = validate(self.payload())
        self.assertEqual(image.size, (24, 16))
        self.assertEqual(image.getpixel((0, 0)), (227, 173, 120))
        self.assertEqual(question, 'Where is the object?')
        self.assertEqual(choices, ['left', 'right'])

    def test_invalid_choice_sets_are_rejected(self):
        for choices in [[], ['only'], ['same', 'same'], ['left', ''], list(map(str, range(13)))]:
            row = self.payload()
            row['candidates'] = choices
            with self.subTest(choices=choices), self.assertRaises(ValueError):
                validate(row)

    def test_corrupt_images_are_rejected(self):
        for value in ['data:image/png;base64,!!!', 'file:///etc/passwd', 'data:image/png;base64,YWJj']:
            row = self.payload()
            row['image'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate(row)

    def test_twelve_user_choices_are_supported(self):
        row = self.payload()
        row['candidates'] = [f'choice {i}' for i in range(12)]
        self.assertEqual(len(validate(row)[2]), 12)


if __name__ == '__main__':
    unittest.main()
