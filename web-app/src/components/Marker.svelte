<script>
  //  Data props 
  export let value = 0
  export let index = 0
  //  Function props
  export let selectMarker
</script>

<button 
    on:click={() => selectMarker(index)}
    class="p-5 lg:p-10 flex justify-center items-center
    {index > 2 ? 'border-white border-t-8' : ''}
    { (((index + 1) % 3) == 1) ? 'border-white border-r-8' : ''}
    { (((index + 1) % 3) == 2) ? 'border-white border-r-8' : ''}
    {value == 0 ? 'hover:bg-white cursor-pointer' : ''}
  ">
  {#if value == 1}
    <!-- X for Player 1 -->
    <svg 
      class="w-8 h-8 text-white md:w-15 md:h-15 lg:w-20 lg:h-20" 
      fill="currentColor" 
      viewBox="0 0 20 20" 
      xmlns="http://www.w3.org/2000/svg">
      <path fill-rule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clip-rule="evenodd"></path>
    </svg>
  {:else if value == 2}
    <!-- O for Player 2 -->
    <svg 
      class="w-8 h-8 text-white md:w-15 md:h-15 lg:w-20 lg:h-20" 
      fill="none" 
      stroke="currentColor" 
      viewBox="0 0 24 24" 
      xmlns="http://www.w3.org/2000/svg">
      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z">
    </svg>
  {:else}
    <div 
      class="w-8 h-8 md:w-15 md:h-15 lg:w-20 lg:h-20">
    </div>
  {/if}
</button> 