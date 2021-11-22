# deep-fish
Experimenting with deep learning for fish ID.

## Command line interface

Create an RLS species dataset:

    $ python manage.py create-rls-species-dataset \
        --m1-csv-path ~/projects/fish-id/data/dump-20210717/m1.csv \
        --image-dir ~/projects/yanirs.github.io/tools/rls/img \
        --output-dir data/rls-species-25-min-images-3/ \
        --num-species 25 \
        --min-images-per-species 3

Create an RLS genus dataset:

    $ python manage.py create-rls-genus-dataset \
        --image-dir ~/projects/yanirs.github.io/tools/rls/img \
        --output-dir data/rls-top-5-genera \
        --num-top-genera 5

Create a test dataset from a trip directory:

    $ python manage.py create-test-dataset \
        --trip-dir ~/Pictures/202010\ Eviota\ GBR \
        --output-dir data/eviota-202010
