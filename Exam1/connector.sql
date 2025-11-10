-- Aggregation with HAVING
SELECT beer, SUM(sales) AS total
FROM sells
GROUP BY beer
HAVING SUM(sales) > 100;

-- LEFT JOIN and COALESCE
SELECT b.name, COALESCE(SUM(s.price),0) AS rev
FROM bars b
LEFT JOIN sells s ON s.bar = b.name
GROUP BY b.name
ORDER BY rev DESC
LIMIT 5;

-- EXISTS subquery
SELECT d.name
FROM drinkers d
WHERE EXISTS (
  SELECT 1 FROM likes L
  JOIN sells S ON S.beer = L.beer
  WHERE L.drinker = d.name AND S.price <= 5
);
