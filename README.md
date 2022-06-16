# salary-prediction-app

Steps to upgrade database changes:

**Locally**
<ul>
<li> python -m flask db init </li>
<li> python -m flask db migrate -m 'Commit Message'</li>
<li> python -m flask db upgrade</li>
</ul>

***Heroku**
<ul>
<li>run heroku run bash after pushing the changes</li>
<li>Then run the above three commands in the heroku bash</li>
</ul>

You actually want to do flask db init and flask db migrate locally. Commit the results to your git repo. Then on Heroku, you only do the heroku run flask db upgrade. <br>[Link](https://stackoverflow.com/a/46430117/14204371)
<br>

[Tutorial YT video](https://youtu.be/ca-Vj6kwK7M)