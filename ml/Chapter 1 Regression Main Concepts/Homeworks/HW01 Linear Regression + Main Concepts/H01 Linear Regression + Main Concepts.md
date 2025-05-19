# Տնային

կցում եմ ֆայլ (`data_lin_reg.csv`) որտեղ երկու սյուն կա՝ x ու y։ 

Պետք ա գտնենք `theta_1 * x` ուղիղը որ լավագույնը նկարագրում ա տվյալները։

Ամենինչ ձեռքով ա պետք գրել՝ կարելի ա օգտվել մենակ մինչև էս պահը անցած գրադարաններից։

1. Դատան քաշեք, կարդացեք պանդասով, վիզուալիզացրեք
2. Gradient descent-ը իմպլեմենտացիա արեք։ Բզբզալով գտեք լավագույն alpha-ն (step size / learning rate)
3. Տարբեր alpha-ների համար գծեք գրաֆիկ որը ցույց կտա իտերացիայի քանակի ու Ռիսկի (միջին սխալանքի) կապը։
4. Վերջում գծեք դատան - լավագույն մոդելի գուշակած կետերը, գուշակած ուղիղը ու իսկական արժեքները։
5 4 3 2 1
## Լրացուցիչ՝
- գծեք սխալանքների չափերի հիստոգրամը։
- գծեք `Theta_1`-ի ու `Risk`-ի կապը արտահայտող գրաֆիկը։ 
  - Լրացուցիչ^2 էդ գրաֆիկի վրա նշեք գրադիենտի հետագիծը։


Եթե մարտահրավերի պատրաստ եք էս ամենինչը արեք ոչթե `theta_1*x` մոդելի համար այլ `theta_0 + theta_1*x`-ի համար

Երբ ստանաք գուշակած գիծը, ուղարկեք բանաձևը՝ ուրախանամ։

# Լրացուցիչ տնային՝ 

դուրս բերել L2 (Mean Square error) Loss-ով լավագույն լուծման անալիզիտ տեսքը (normal equation-ա կոչվում)

Գործնականի ժամանակ ամենայն հավանակնությամբ չենք նայի դրա համար միանգամից ուղարկեմ լրացուցիչ նյութեր, նաև լուծումը։ Բայց առանց մտածելու լուծումը նայելը շատ վատ միտք ա։

"""
- https://www.youtube.com/watch?v=NN7mBupK-8o it's quite old but I think it does a good job explaining it. Here J(theta) is just the same as R_emp in our notation
- And this is also a nice post http://mlwiki.org/index.php/Normal_Equation 
- And if you want to see the derivation of the formula, this forum is quite helpful https://math.stackexchange.com/questions/4177039/deriving-the-normal-equation-for-linear-regression 
"""