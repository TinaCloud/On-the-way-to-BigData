# SQL (Structured Query Language) 

## SQL-1
### Datatypes

- varchar
- int
- float


### Database and tables
### create database

create database command
```
-- CREATE DATABASE db_name;
-- The '--' or '#' indicate a comment. Comments are not executed.
-- You won't be able to create new databases, but you'll use the one we created for you.
```
all MYSQL commands must end with ;

```
SHOW DATABASES;
-- You should see the database you were assigned from the <strong>Courseware section</strong>.
```

Once the database has been created, execute the following command to use it.
```
USE db_name; 
-- Check your dashboard and use the db_name you were assigned.
```
We can use the Select command to show which database we're currently using.
```
SELECT database();
```
you can also execute multiple lines by highlighting a block, and including a shift in the mix (so Shift + Control + Enter or Shift + Command + Enter).

```
SHOW DATABASES;        
-- see which database is available.

SELECT current_user(); 
-- See the current user you're connected as.

SELECT database();     
-- See what database we're using now.
```
A new table can be created using the CREATE TABLE command:
```
CREATE TABLE table_name
(
  column_name1 data_type(size) constraint,
  column_name2 data_type(size),
...
);
```

Example 1:
```
CREATE TABLE Student
(
  student_id int(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
  first_name varchar(25) NOT NULL,
  last_name varchar(25) NOT NULL,
  grade_level tinyint(5) NOT NULL
);
```
Constraints:
- NULL
	- NULL is used to indicate the absence of a value. Therefore, NOT NULL is a constraint which says we do not want the field to be NULL.
- UNIQUE
	- In a UNIQUE constraint, the values in the columns should be unique. For example, each car in the Unites States is assigned to a (VIN) Vehicle Identification Number. To check no two rows in a VIN column holds the same value, a unique constraint can be added to the column. Unique constraints can be added to one or multiple columns when defining a query.
- SIZE
	- The type(size) indicates the type and space each entry is going to take. For example, if you do varchar(26), you're stating that the maximum size of entry is 26 characters. Anything longer will be truncated. We specify this because of how databases are designed for efficiency. You want to specify something larger than you'd need, but not too large.
- Unsigned
	- it is only positive

Once the table structure has been created we can use SHOW TABLES to see updated tables in the database School.
```
SHOW TABLES;
```
To see the layout of the table use DESCRIBE tablename. This shows only parameters, no data is shown and neither is how the table is populated.
```
DESCRIBE Student;
```
### Inserting into a table

```
INSERT INTO tablename
  (column_name_1,column_name_2...)
  VALUES ('value1','value2'...);
```
Notes:
- MYSQL uses singles quotes to indicate the beginning and end of a character.
- In the case where there are quotes for a value, double quotes can be used.

What if you want to insert multiple rows? Is it possible? Try it.
```
INSERT INTO Student(first_name,last_name,grade_level)
   VALUES('Fred','Doe',11),
         ('Lebron','James',12),
         ('Michael','Franklin',11),
         ('Robert','Stark',9);
-- Note the lack of quotes around the grade_levels, since they're numbers we don't need to quote them.
-- It will still work with quotes, since MySQL will try its best to convert/cast to proper type.
```

### select from a table
to display all rows in the table without any conditions use the command:
```
SELECT * FROM tablename;
```
Notes:
-* means every column

What if we're interested in only a subset of columns?

To display all values from the first_name column we use the command SELECT column_name FROM tablename;

```
-- Make sure you have the Student table still.
SHOW TABLES;
DESCRIBE Student; -- Take a peek at the table.
SELECT first_name FROM Student;
```
Further Filtering

```
SELECT * FROM tablename WHERE some_condition;
```
Example
```
SELECT * FROM Student WHERE first_name='Michael';
```
```
SELECT * FROM TestScore WHERE test_name='AI Final 2' AND student_id=3;
```

### Overview of Joins
The FOREIGN KEY reference is indicating that the column is related to (and referencing) another Table.

Logically speaking, each row inside the TestScore table should belong to a specific row in Student. If in the TestScore table we wanted to include the information from the Student table, we could simply add additional columns for first_name,last_name, etc. But it's repetitive since we already have a table with each row representing a student. So instead **we include an id (FOREIGN KEY TestScore.student_id) that references the Student table.** Don't worry about its exact technical detail, but know that it REFERENCES the Student table.

## Join
Joins are used to retrieve data from multiple tables.

You're essentially combining 2 tables, based on a rule that matches certain columns.

So you'd think it'd be simple to join 2 tables together, right? Well, not exactly.

The tricky part to specify is how you're going to deal with entries that are in one table, but not the other. There is actually quite a spectrum of options when doing joins if you take a quick look at the image below. The range of possible types of joins are on a spectrum: on one side, you can retain only those entries that are in both tables (intersection). On the other side, you can retain all entries that are in either (union).

![SQL Joins](http://i.imgur.com/ABgpLNh.png)

As the image might suggest,** we choose one column from both tables and match them for join.** You can actually join on multiple columns, but let's ignore that for now.

### Inner Join
Inner Joins (which is the same as simply a Join) displays only the matching records from both tables and is the most common type of join.

![Inner Join](http://i.imgur.com/hFo1eOZ.png)

Q: how we're getting the intersection of the tables is via a column. 

Example:
```
SELECT column1,column2... FROM Table_1
INNER JOIN Table_2
ON Table_1.column_name=Table_2.column_name;
```

An important thing to note that the 1st table that gets mentioned is considered the LEFT table, and the 2nd table is the RIGHT table. This will become important next module, when we discuss Outer Joins.

1. Destroy your existing tables: 

```
SELECT database(); -- See what database we're using now.
DROP TABLE IF EXISTS TestScore; -- ** See below why we need to drop this.
DROP TABLE IF EXISTS Student;
DROP TABLE IF EXISTS Club;
-- Now you can run the code that creates the 2 tables.

```
Notes:
We need to drop the TestScore table, since we included a FOREIGN KEY constraint that REFERENCES the Student table. That means if we were to drop/destroy the student table, we would violate the rule (since then the reference would not make sense). So we drop the TestScore table first.

2. Create new tables:

```
CREATE TABLE Student
(
  student_id int(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
  first_name varchar(25) NOT NULL,
  last_name varchar(25) NOT NULL,
  grade_level tinyint(5) NOT NULL
);
 
CREATE TABLE Club
(
  club_id int(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
  student_id int(10) UNSIGNED NOT NULL,
  birthdate DATE NOT NULL,
  role varchar(25) NOT NULL,
  years_served tinyint(5) NOT NULL
  -- We're not going to include the foreign key clause here, so we don't have a hard constraint.
);

SHOW TABLES;
```

3. Insert values

```
-- First Student
INSERT INTO Student(first_name, last_name, grade_level)
VALUES('Fred', 'Allen',9),
     ('Allen','Gates',12),
     ('Paula','Jobs',10),
     ('Benjamin','Franklin',10),
     ('Tony','Stark',10),
     ('Steve','Rogers',11),
     ('Marge','Simpson',12),
     ('Bart', 'Simplson',10)
;
 
-- Now Club
INSERT INTO Club(student_id, birthdate, role, years_served)
VALUES(1,'2001-07-09','greeter',1), -- For date use YYYY-MM-DD
     (2, '1997-05-22','president',4),
     (3,'2000-09-17','treasurer',2),
     (4,'2000-02-14','vice president',2),
     (9, '2001-12-14', 'member',1),
     (10,'1998-10-10','member',1),
     (11,'1997-01-06','secretary',3),
     (12,'2000-02-13','member',3)
;

SELECT * FROM Club;

```
4. Inner Join
```
-- EXECUTE THIS
SELECT first_name,last_name FROM Student
INNER JOIN Club
ON Student.student_id = Club.student_id;
```
You can also combine a condition with join:
```
SELECT * FROM Student
INNER JOIN Club
ON Student.student_id = Club.student_id
WHERE years_served > 1; 
```

### (Left/Right) Outer Join

- LEFT JOIN
- RIGHT JOIN

When deciding which join to execute, essentially what it comes down to is deciding how you want to deal with rows that don't match on the key you defined the join by. For Inner Join, we ignored ALL mismatched rows.

So for Outer Joins, we display all values on one side (the side that you picked). We try to match it as best as we can, and fill unmatched rows with NULLs.

1. LEFT Outer JOIN = LEFT JOIN

LEFT OUTER JOIN displays all the records from the left table plus the intersecting records.

![Left Join](http://i.imgur.com/gxkao9k.png)

Example

```
SELECT column1,column2... -- or *
FROM Table_A              -- Remember that the first table mentioned is the LEFT table.
LEFT OUTER JOIN Table_B
ON table1.column_name = table2.column_name;
```

Left outer join example:

```
SELECT * FROM Student
LEFT OUTER JOIN Club
ON Student.student_id=Club.student_id;
```
Notes:
- In the above question, Student is the left table because it is lying on the left side of the statement (when they're all put on one line).
- Every entry from the original Student Table is present in the output. This is verified by seeing that every Student.student_id is available in the end product. 
	- When there isn't a corresponding entry for a Student row in Club, there are NULLs.
- Entries from Club is joined as best as possible, but those Club rows without a corresponding Student row is omitted.

2. Right outer join example:
```
SELECT * FROM Student
RIGHT OUTER JOIN Club
On Student.student_id=Club.student_id;
```

3. FULL OUTER JOIN
There isn't actually a direct command for Full Outer Join in MySQL, but it's essentially a combination of both tables (nulls on both sides).

![Outer Join](http://i.imgur.com/It7zttg.png)

FULL OUTER JOIN is essentially a Join that includes ALL records, including NULLs with unmatched records.

### Join conclusion
Inner Joins are the most common type of join. Outer Joins (right outer join or left outer join) can be used to include in the join right or left table even with no matching.

The Right Outer Join includes all rows from the right table as well as matching records.

The Left Outer Join includes all rows from the left table as well as matching records.

# Check
JOIN quiz failed in the MySQL workbench for the following codes:

the reason is that the access to the database Shared is denied by using USE Shared
```
SELECT * FROM Shared.Movie
INNER JOIN Shared.Rating
On Shared.Movie.movie_id=Shared.Rating.movie_id;
```

## SQL-2
### SQL Scripts Intro
The idea is that you want to execute an entire sql script via the SOURCE command.

```
# In Terminal / Bash
# When you're in a mysql console client, you can run the following:
mysql> source /Users/[username]/Desktop/scripts/analysis.sql;
# The /Users/... is the path to the sql file you want to execute.
```
### Import a CSV
```
-- The first line is the header, and is optional.
first_name,email,age
john,john@email.com,39
jane,jane@email.com,28
```

The general steps are as follows:

- Decide what kind of columns (int(10), int(8), varchar(26) etc) you want to use for your columns (if you don't specify a size, a default one will be assigned).
- Create the Database that fits the above specs.
- Run the LOAD DATA command to do the import into table.
- If you encounter any error and have to redesign your database, DROP the table and repeat the previous steps.

download the tags.csv file
```
-- Create the table that you will import into.
CREATE TABLE MyTag
(
  rating_id   int(10) UNSIGNED AUTO_INCREMENT PRIMARY KEY NOT NULL,
  user_id     int(10) UNSIGNED NOT NULL,
  movie_id    int(10) UNSIGNED NOT NULL,
  tag         varchar(128) NOT NULL,
  time_tagged int(11) NOT NULL
);
```
If the above table creation didn't satisfy the requirement (say LOAD DATA below throws an error), you can destroy the table by:

```
DROP TABLE MyTag; 
-- Do this only if you need to recreate the table
```
Now you load the data 
```
-- Update the path and execute the query.
LOAD DATA LOCAL INFILE '/Users/t-rex-Box/data/tags.csv' -- this string is where the file is on your computer.
INTO TABLE MyTag                       -- Name of the table you created earlier
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\r\n'             -- You want to change this to '\r' if you're on a MAC!
IGNORE 1 LINES                         -- This is for the header
(user_id, movie_id, tag, time_tagged); -- name of the matching columns
```
The LOCAL keyword tells it to go through your local computer, locate the local file, and upload it to our cloud database.

Q: How many rows resulted in MyTag table? 
```
SELECT COUNT(*) FROM MyTag;
```
**Failed in loading the files, no errors but no rows affected
**

### Aggregate Functions Intro
When performing aggregations AVG, SUM, MAX and MIN will ignore NULL values. However, COUNT includes the row with the NULL value.

```
-- Example
SELECT AGGREGATE(column_name) FROM table_name;
```
So let's say we want to aggregate the Average Rating from the rating database. The first thing we do is describe it.
```
DESCRIBE Shared.Rating;

SHOW Databases;
-- Make sure you have the Shared database.
SELECT AVG(rating) FROM Shared.Rating;
```

Q: What is the movie_id for the movie titled: 'GoldenEye'? in Shared.Movie
```
SELECT * FROM Shared.Movie WHERE title='GoldenEye';
```

Q: What is the average rating for 'Golden Eye'?
```
SELECT AVG(rating) FROM Shared.Rating WHERE movie_id=10;
```
Q: What is the year of the oldest movie?
```
SELECT min(year_released) FROM Shared.Movie;
```

### Group By
So what if you want to apply an aggregate function to a few different movies?

The steps you want to repeat are as follows:

- Find out the movie_id for each of the movie you want to find the average rating.
- For each of those movie_id, take its respective subset from Rating, and find the AVG for for it.

The above operation can be done via a Group By operation.

**Big Idea**
GROUP BY statements allow you to bucket via a column, and apply an aggregate function.

Here is a visualization of what happens.

![Group By](http://.png)

1. We identify the unique makers and bucket them into their respective chunks via GROUP BY. We use the unique maker values as the bucket identifier.

2. For each of the individual bucket, we apply an aggregate function (in this case, AVG).

3. The result is packaged into a single table.

Application: GROUP BY Aggregation on our dataset.
```
-- First let's see what's in the table.
DESCRIBE Shared.Rating; 

-- Now for the aggregation.
SELECT movie_id, AVG(rating) FROM Shared.Rating GROUP BY movie_id;
```

Write the statement that returns the number of ratings that a user has submitted, for ALL users. 
```
SELECT  user_id, count(rating) FROM Shared.Rating GROUP BY user_id;
```

### Introducing Subquery
A Subquery is a nested query used within a SELECT, INSERT, DELETE or UPDATE statement. It can also be used within another subquery, though that combination is rarer.
```
-- Format 1:
(Query 2(Query 1))

-- Format 2:
 Query3(Query 2(Query 1)))
```

The Subquery (also inner query) is Query 1 and is executed first. The outer query or container statement (Query 2) is executed next.

In the case where there are multiple subqueries. Query 1 and Query 2 would be the inner queries and Query 3 would be the container statement.

So going back to the Shared.Rating table, let's say we want to find out all movies that received a rating that's higher than average.
```
-- We know that the following will yield the average rating for all movies.
SELECT AVG(rating) from Shared.Rating;
 
SELECT movie_id, rating FROM  Shared.Rating
WHERE rating > (SELECT AVG(rating) from Shared.Rating);
```
Q: How many movies had a rating that was lower than the average?
```
SELECT COUNT(movie_id) FROM Shared.Rating 
WHERE rating < (SELECT AVG(rating) from Shared.Rating)
;
```
### More Joins -- focusing on INNER JOIN
Example:
```
SELECT * FROM Shared.Rating
INNER JOIN Shared.Movie
ON Shared.Rating.movie_id=Shared.Movie.movie_id;
```

Now let's say you want to find the average rating for a movie called 'Pulp Fiction' (you might've heard about it).

The steps we want to take is as follows:

- Create a joined table for Movie | Rating with just the movie 'Pulp Fiction'. Use JOIN and WHERE for this.

- Apply the aggregate AVG() to the result. Use subqueries for this.

```
-- First just to get the joined table with 'Pulp Fiction' ratings:

SELECT count(title), title, rating FROM Movie
INNER JOIN Rating
On Rating.movie_id = Movie.movie_id
WHERE Movie.title = 'Pulp Fiction'
;
```

### Alias
we JOINED the Movie and Rating table and made a subset WITHOUT naming that resulting table. As a result, there is no way of utilizing that table. The error was stating that we need to alias (i.e. name) that table, allowing us to access it in the outer query:

```
-- Getting rating from the table.
SELECT PulpFictionTable.title, AVG(PulpFictionTable.rating) FROM -- Notice the aliased name!
(
  SELECT title, rating FROM Movie
  INNER JOIN Rating
  On Rating.movie_id = Movie.movie_id
  WHERE Movie.title = 'Pulp Fiction'
) as PulpFictionTable                                            -- This is our alias for the joined table
;
```
We access the PulpFictionTable's column via the original column name. We can renamed it as follows:
```
-- Renaming all around
SELECT PulpFictionTable.PF_title, AVG(PulpFictionTable.PF_rating) FROM  -- Using renamed Table + renamed Columns
(
  SELECT title as PF_title, rating as PF_rating FROM Movie              -- Renaming the Columns as well!
  INNER JOIN Rating
  On Rating.movie_id = Movie.movie_id
  WHERE Movie.title = 'Pulp Fiction'
) as PulpFictionTable                                                   -- Aliasing the table again.
;
```

### Join Exercise
```
SELECT PuplFictionRating.PF_title, AVG(PuplFictionRating.PF_rating) FROM
(
  SELECT title as PF_title, rating as PF_rating FROM Movie
  INNER JOIN Rating
  On Rating.movie_id = Movie.movie_id
  WHERE Movie.title = 'Pulp Fiction'
) as PuplFictionRating
;
```
Q: It yields a table that shows all unique movies and their respective average ratings?
```
SELECT Movie.title, AVG(Rating.rating) as avg_rating FROM Movie -- this aliasing will be used later.
INNER JOIN Rating
On Movie.movie_id=Rating.movie_id
GROUP BY Movie.movie_id
;
```
**Why from Movie??**

## ProblemSet
```
USE ProblemSet;
Describe Concession;

Select COUNT(concession_id) FROM Concession;

Select concession_id, order_create_time FROM Concession 
ORDER BY order_create_time ASC;

SELECT category_name, sum(revenue) FROM Concession
Group by category_name
ORDER BY sum(revenue) DESC;

SELECT category_group, sum(revenue) FROM Concession
Group by category_group
ORDER BY sum(revenue) DESC;

# Part 3
Select category_name, avg(revenue) FROM Concession
Where category_name='beer' # from last question we know it is 'beer'
Group by category_name;

SELECT category_name, avg(revenue) FROM Concession
Group by order_create_time
ORDER BY avg(revenue) DESC;

# What category_name had the highest per-item revenue (you'll have to take into account the quantity)?

do not understand
```

Notes:
- Functions include Database, table, schema, data type, primary key, operator, wildcard, search pattern, field, aggregate function, join, union, insert, update, view, execute, transaction process, rollback, commit, savepoint, cursor, constraint.
- No difference between capital and lower case

> Select prod_id, RTRIM(prod_name) + ‘(‘ + RTRIM(vend_country) +’)’ as vend_title, quantity*item_price # RTRIM() delete all the space on the right.
From products
Where (prod_id = ‘DL’ or prod_id = ‘va’) and prod_price !< 3.49
Where prod_id in (‘DL’,’va’)
Where not prod_id = ‘DL’
Where prod_name like ‘fish%’ # ‘%’ means any character any time, ‘_’ means single character, ‘[]’ means 
Group by vend_id
Having count(*) >=2

- “where” filter cols, but “having” filters groups
- filter condition, where prod_price between 5 and 10.
- the reason why use () is that the priority of “and” is higher than “or”. () is preferred anytime when “and” and “or” are used together.

>Order by prod_name, prod_price DESC (where asc is default);
(order by 2, 3 means by the 2nd column, 3rd column)

-
>Update customers
Set cust_contact = ‘Sam’
Cust_email = ‘Email’
Where cust_id = ‘007’

Another example
> ![sql example](http://i.imgur.com/Vozfw1b.jpg)

![sql example](http://i.imgur.com/PKgKHft.png)

> ALTER TABLE tablename
(
  ADD|DROP  column  datatype  [NULL|NOT NULL]  [CONSTRAINTS],
  ADD|DROP  column  datatype  [NULL|NOT NULL]  [CONSTRAINTS],
    ...
);

% ALTER TABLE is used to update the schema of an existing table. To create a new table, use CREATE TABLE. 
> CREATE TABLE tablename
(
    column    datatype    [NULL|NOT NULL]    [CONSTRAINTS],
    column    datatype    [NULL|NOT NULL]    [CONSTRAINTS],
       ...
);

> COMMIT [TRANSACTION];

% COMMIT is used to write a transaction to the database.
 
> CREATE INDEX indexname
ON tablename (column, ...);

%CREATE INDEX is used to create an index on one or more columns. 
> CREATE PROCEDURE procedurename [parameters] [options]
AS
SQL statement;

% CREATE PROCEDURE is used to create a stored procedure. 

> CREATE VIEW viewname AS
SELECT columns, ...
FROM tables, ...
[WHERE ...]
[GROUP BY ...]
[HAVING ...];

% CREATE VIEW is used to create a new view of one or more tables.
 
> DELETE FROM tablename
[WHERE ...];

% DELETE deletes one or more rows from a table. 

> DROP INDEX|PROCEDURE|TABLE|VIEW
indexname|procedurename|tablename|viewname;

% DROP permanently removes database objects (tables, views, indexes, and so forth). 

> INSERT INTO tablename [(columns, ...)]
VALUES(values, ...);

% INSERT adds a single row to a table. 

> INSERT INTO tablename [(columns, ...)]
SELECT columns, ... FROM tablename, ...
[WHERE ...];

% INSERT SELECT inserts the results of a SELECT into a table.

> ROLLBACK [ TO savepointname];
> 
% ROLLBACK is used to undo a transaction block. See Lesson 20 for more information.

> SELECT columnname, ...
FROM tablename, ...
[WHERE ...]
[UNION ...]
[GROUP BY ...]
[HAVING ...]
[ORDER BY ...];

% SELECT is used to retrieve data from one or more tables (or views).
 
> UPDATE tablename
SET columname = value, ...
[WHERE ...];

% UPDATE updates one or more rows in a table. 





