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

<br>

[Tutorial YT video](https://youtu.be/ca-Vj6kwK7M)