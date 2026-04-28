# veracity-emotions
# Andrew Clark, Scott Landry, Jason Ossai
COMP4415 group project: **train an emotion model on GoEmotions**, apply it to **Twitter15/16 source tweets**, and predict **true vs false rumors** using **emotional features**—plus a **text baseline** on the same splits to show whether emotion adds signal beyond generic wording.

## To run setup.sh:

Make sure to install pip and python 3.12> first.

Use `setup.sh` from the project root so all relative paths work correctly.

```
cd /path/to/veracity-emotions
chmod +x setup.sh
./setup.sh
```

Python version:
- The script automatically uses `python3` if available, otherwise it falls back to `python`.
- Make sure at least one Python command works first: `python3 --version` or `python --version`.

## To run text model:
```
python(3) {PROJECT_ROOT}/src/text_model.py
```


## manual download
```     
Twitter15/16:
https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=1
GoEmotions:
https://www.kaggle.com/datasets/debarshichanda/goemotions
## References
```
```
[1] X. Liu, A. Nourbakhsh, Q. Li, R. Fang, and S. Shah. Real-time Rumor Debunking on Twitter.
Proceedings of the 24th ACM International Conference on Information and Knowledge
Management, pages 1867–1870, 2015.

[2] J. Ma, W. Gao, P. Mitra, S. Kwon, B. J. Jansen, K.-F. Wong, and C. Meeyoung. Detecting Rumors
from Microblogs with Recurrent Neural Networks. Proceedings of the 25th International Joint
Conference on Artificial Intelligence, 2016.

[3] J. Ma, W. Gao, and K.-F. Wong. Detect Rumors in Microblog Posts Using Propagation Structure
via Kernel Learning. Proceedings of the 55th Annual Meeting of the Association for Computational
Linguistics (ACL), 2017.

[4] D. Demszky, D. Movshovitz-Attias, J. Ko, A. Cowen, G. Nemade, and S. Ravi. GoEmotions: A
Dataset of Fine-Grained Emotions. Proceedings of the 58th Annual Meeting of the Association for
Computational Linguistics (ACL), 2020.
```