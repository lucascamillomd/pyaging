document.addEventListener('DOMContentLoaded', function() {
    // Add filter controls above the table
    const table = document.querySelector('.sortable.filterable');
    if (!table) return;
    
    const filtersDiv = document.createElement('div');
    filtersDiv.className = 'glossary-filters';
    
    // Create data type filter
    const dataTypes = [...new Set(Array.from(table.querySelectorAll('tbody tr td:nth-child(2)')).map(cell => cell.textContent.trim()))];
    const dataTypeFilter = createFilter('Data type', dataTypes);

    // Create species filter
    const species = [...new Set(Array.from(table.querySelectorAll('tbody tr td:nth-child(3)')).map(cell => cell.textContent.trim()))];
    const speciesFilter = createFilter('Species', species);
    
    // Create search box
    const searchDiv = document.createElement('div');
    const searchLabel = document.createElement('label');
    searchLabel.textContent = 'Search:';
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'Type to search...';
    searchDiv.appendChild(searchLabel);
    searchDiv.appendChild(searchInput);
    
    // Add filters to the container
    filtersDiv.appendChild(dataTypeFilter);
    filtersDiv.appendChild(speciesFilter);
    filtersDiv.appendChild(searchDiv);
    
    // Insert the filters before the table
    table.parentNode.insertBefore(filtersDiv, table);
    
    // Link DOIs
    table.querySelectorAll('tbody tr td:nth-child(5)').forEach(cell => {
      const doiText = cell.textContent.trim();
      if (doiText.startsWith('http')) {
        cell.innerHTML = `<a href="${doiText}" target="_blank">${doiText}</a>`;
      }
    });
    
    // Make the table headers sortable
    table.querySelectorAll('th').forEach(header => {
      header.addEventListener('click', () => {
        const index = Array.from(header.parentNode.children).indexOf(header);
        sortTable(table, index);
      });
    });
    
    // Add filtering functionality
    const filters = filtersDiv.querySelectorAll('select');
    filters.forEach(filter => {
      filter.addEventListener('change', () => applyFilters());
    });
    
    searchInput.addEventListener('input', () => applyFilters());
    
    // Initial sort by the first column
    sortTable(table, 0);
    
    function createFilter(label, options) {
      const div = document.createElement('div');
      const labelEl = document.createElement('label');
      labelEl.textContent = label + ':';
      const select = document.createElement('select');
      
      // Add default "All" option
      const defaultOption = document.createElement('option');
      defaultOption.value = '';
      defaultOption.textContent = 'All';
      select.appendChild(defaultOption);
      
      // Add data options
      options.sort().forEach(option => {
        const optionEl = document.createElement('option');
        optionEl.value = option;
        optionEl.textContent = option;
        select.appendChild(optionEl);
      });
      
      div.appendChild(labelEl);
      div.appendChild(select);
      return div;
    }
    
    function applyFilters() {
      console.log("Applying filters..."); 
      // Alternative way to select the dropdowns
      const dataTypeSelect = filtersDiv.children[0].querySelector('select'); 
      const speciesSelect = filtersDiv.children[1].querySelector('select');  
      
      const dataType = dataTypeSelect.value;
      const species = speciesSelect.value;
      const searchText = searchInput.value.toLowerCase();
      console.log(`Filters - DataType: '${dataType}', Species: '${species}', Search: '${searchText}'`);

      table.querySelectorAll('tbody tr').forEach(row => {
        const rowDataType = row.querySelector('td:nth-child(2)').textContent.trim();
        const rowSpecies = row.querySelector('td:nth-child(3)').textContent.trim();
        const rowText = row.textContent.toLowerCase();
        
        const dataTypeMatch = dataType === '' || rowDataType === dataType;
        const speciesMatch = species === '' || rowSpecies === species;
        const searchMatch = searchText === '' || rowText.includes(searchText);

        // Log matching status for the first few rows to see what's happening
        // (Remove this or limit it after debugging)
        if (Array.from(table.querySelectorAll('tbody tr')).indexOf(row) < 5) {
             console.log(`Row ${Array.from(table.querySelectorAll('tbody tr')).indexOf(row)} - Data: '${rowDataType}', Species: '${rowSpecies}' -> Matches: D=${dataTypeMatch}, S=${speciesMatch}, T=${searchMatch}`);
        }
        
        row.style.display = (dataTypeMatch && speciesMatch && searchMatch) ? '' : 'none';
      });
    }
    
    function sortTable(table, columnIndex) {
      const header = table.querySelectorAll('th')[columnIndex];
      const currentSort = header.getAttribute('aria-sort');
      
      // Remove sort indicators from all headers
      table.querySelectorAll('th').forEach(h => h.removeAttribute('aria-sort'));
      
      // Determine sort direction
      const sortDirection = currentSort === 'ascending' ? 'descending' : 'ascending';
      header.setAttribute('aria-sort', sortDirection);
      
      // Get table rows and sort
      const tbody = table.querySelector('tbody');
      const rows = Array.from(tbody.querySelectorAll('tr'));
      
      rows.sort((a, b) => {
        const aText = a.querySelectorAll('td')[columnIndex].textContent.trim();
        const bText = b.querySelectorAll('td')[columnIndex].textContent.trim();
        
        // Try to convert to numbers if possible
        const aNum = parseFloat(aText);
        const bNum = parseFloat(bText);
        
        if (!isNaN(aNum) && !isNaN(bNum)) {
          return sortDirection === 'ascending' ? aNum - bNum : bNum - aNum;
        }
        
        // Fall back to string comparison
        return sortDirection === 'ascending' 
          ? aText.localeCompare(bText) 
          : bText.localeCompare(aText);
      });
      
      // Re-append rows in sorted order
      rows.forEach(row => tbody.appendChild(row));
    }
  });