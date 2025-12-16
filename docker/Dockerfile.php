FROM php:8.2-apache

# Install PHP extensions
RUN docker-php-ext-install pdo pdo_mysql

# Enable Apache mod_rewrite
RUN a]2enmod rewrite

# Install Composer
COPY --from=composer:latest /usr/bin/composer /usr/bin/composer

# Set working directory
WORKDIR /var/www/html

# Copy application files
COPY . .

# Install dependencies
RUN composer install --no-dev --optimize-autoloader || true

# Create storage directories
RUN mkdir -p storage/scans storage/rate_limits \
    && chown -R www-data:www-data storage \
    && chmod -R 755 storage

# Apache configuration
RUN echo '<Directory /var/www/html>\n\
    Options Indexes FollowSymLinks\n\
    AllowOverride All\n\
    Require all granted\n\
</Directory>' > /etc/apache2/conf-available/oralcare.conf \
    && a2enconf oralcare

# Expose port
EXPOSE 80

CMD ["apache2-foreground"]
